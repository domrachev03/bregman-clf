"""MJCF -> torque-mode conversion.

Replaces whatever actuators the source MJCF ships (e.g. the Menagerie iiwa14's PD servos) with one
`<motor gear="1">` per hinge/slide joint, clipped to `tau_max`, so the model is driven by raw joint
torques. Joint `damping` and `frictionloss` are zeroed, on joint instances and on every `<default>`
class; `armature` and the joint ranges are left untouched.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

_ACTUATED_JOINT_TYPES = ("hinge", "slide")


def _atomic_write_xml(tree: ET.ElementTree, dest: Path) -> None:
    """Write `tree` to `dest` atomically (temp file + `os.replace`): parallel sweep workers all
    write the same cache path, and a reader must never see a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _merge_defaults(elem: ET.Element, inherited: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Attrs one `<default>` level contributes per child tag, merged over `inherited`."""
    merged = {tag: dict(attrs) for tag, attrs in inherited.items()}
    for child in elem:
        if child.tag == "default":
            continue
        merged[child.tag] = {**merged.get(child.tag, {}), **child.attrib}
    return merged


def _collect_default_classes(root: ET.Element) -> dict[str, dict[str, dict[str, str]]]:
    """`class name -> {tag: merged attrs}`, walking nested `<default class="...">` (MJCF inheritance)."""
    classes: dict[str, dict[str, dict[str, str]]] = {}
    top = root.find("default")
    if top is None:
        return classes

    def recurse(elem: ET.Element, inherited: dict[str, dict[str, str]]) -> None:
        merged = _merge_defaults(elem, inherited)
        cls = elem.attrib.get("class")
        if cls is not None:
            classes[cls] = merged
        for child in elem.findall("default"):
            recurse(child, merged)

    recurse(top, {})
    return classes


def _joint_type(joint: ET.Element, own_class: str | None, classes: dict[str, dict[str, dict[str, str]]]) -> str:
    """Effective MJCF `type` of a joint instance (unset -> MJCF default `"hinge"`)."""
    if "type" in joint.attrib:
        return joint.attrib["type"]
    if own_class is not None:
        return classes.get(own_class, {}).get("joint", {}).get("type", "hinge")
    return "hinge"


def _iter_joints(root: ET.Element, classes: dict[str, dict[str, dict[str, str]]]) -> list[tuple[ET.Element, str]]:
    """`(joint element, resolved type)` for every `<joint>` under `<worldbody>`, in tree (qpos) order."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")

    out: list[tuple[ET.Element, str]] = []

    def recurse(elem: ET.Element, childclass: str | None) -> None:
        cc = elem.attrib.get("childclass", childclass)
        for child in elem:
            if child.tag == "joint":
                own_class = child.attrib.get("class", cc)
                out.append((child, _joint_type(child, own_class, classes)))
            elif child.tag == "body":
                recurse(child, cc)

    recurse(worldbody, None)
    return out


def _zero_damping_everywhere(root: ET.Element) -> None:
    """`damping=0`, `frictionloss=0` on every `<joint>` tag: instances AND `<default>` classes."""
    for joint in root.iter("joint"):
        joint.set("damping", "0")
        joint.set("frictionloss", "0")


def _rewrite_meshdir_absolute(root: ET.Element, source_dir: Path) -> None:
    """Make `<compiler meshdir=...>` absolute so the converted file is relocatable (out_dir != source dir)."""
    asset = root.find("asset")
    has_meshes = asset is not None and asset.find("mesh") is not None
    compiler = root.find("compiler")
    if compiler is None:
        if not has_meshes:
            return
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    meshdir = compiler.attrib.get("meshdir", ".")
    compiler.set("meshdir", str((source_dir / meshdir).resolve()))


def _set_actuators(
    root: ET.Element,
    joints: list[tuple[ET.Element, str]],
    tau_max: Mapping[str, float] | Sequence[float],
) -> list[str]:
    """Replace `<actuator>` contents with one `<motor gear="1">` per hinge/slide joint."""
    actuated = [(j, t) for j, t in joints if t in _ACTUATED_JOINT_TYPES]
    if isinstance(tau_max, Mapping):
        limits = [float(tau_max[j.attrib["name"]]) for j, _ in actuated]
    else:
        limits = [float(x) for x in tau_max]
        if len(limits) != len(actuated):
            raise ValueError(f"tau_max has {len(limits)} entries, expected {len(actuated)} hinge/slide joints")

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    else:
        for child in list(actuator):
            actuator.remove(child)

    names: list[str] = []
    for (joint, _), tau in zip(actuated, limits, strict=True):
        name = joint.attrib["name"]
        names.append(name)
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-tau:.17g} {tau:.17g}",
            },
        )
    return names


def to_torque_mode(
    mjcf_path: str | Path,
    tau_max: Mapping[str, float] | Sequence[float],
    out_dir: str | Path,
) -> tuple[Path, list[str]]:
    """Write `<out_dir>/<stem>_torque.xml` and return `(path, actuated joint names)`.

    `tau_max` is keyed by joint name or given in joint (qpos) order. The joint order equals the
    actuator order. Mesh paths are kept valid by rewriting `compiler/@meshdir` to an absolute path.
    """
    mjcf_path = Path(mjcf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mjcf_path.stem

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    classes = _collect_default_classes(root)
    joints = _iter_joints(root, classes)

    _zero_damping_everywhere(root)
    _rewrite_meshdir_absolute(root, mjcf_path.parent)
    joint_names = _set_actuators(root, joints, tau_max)

    path = out_dir / f"{stem}_torque.xml"
    _atomic_write_xml(tree, path)
    return path, joint_names
