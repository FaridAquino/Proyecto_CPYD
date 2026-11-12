"""
cpu_topology.py -- Topología híbrida del procesador (Windows).

Usa GetSystemCpuSetInformation (kernel32) para obtener, por CPU lógica:
  EfficiencyClass (mayor = núcleo de rendimiento, P-core; 0 = E-core) y
  CoreIndex (CPUs lógicas con el mismo CoreIndex comparten núcleo físico: SMT).
Imprime la tabla y guarda results/cpu_topology.json.
"""
import ctypes
import json
import os
from ctypes import wintypes


class _CpuSet(ctypes.Structure):
    _fields_ = [("Id", wintypes.DWORD), ("Group", wintypes.WORD),
                ("LogicalProcessorIndex", ctypes.c_ubyte), ("CoreIndex", ctypes.c_ubyte),
                ("LastLevelCacheIndex", ctypes.c_ubyte), ("NumaNodeIndex", ctypes.c_ubyte),
                ("EfficiencyClass", ctypes.c_ubyte), ("AllFlags", ctypes.c_ubyte),
                ("SchedulingClass", wintypes.DWORD), ("AllocationTag", ctypes.c_uint64)]


class _Info(ctypes.Structure):
    _fields_ = [("Size", wintypes.DWORD), ("Type", wintypes.DWORD), ("CpuSet", _CpuSet)]


def topology():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    n = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
    buf = ctypes.create_string_buffer(n.value)
    if not k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
        raise OSError(ctypes.get_last_error())
    out, off = [], 0
    while off < n.value:
        info = _Info.from_buffer_copy(buf.raw[off:off + ctypes.sizeof(_Info)])
        cs = info.CpuSet
        out.append(dict(cpu=cs.LogicalProcessorIndex, core=cs.CoreIndex,
                        eff=cs.EfficiencyClass))
        off += info.Size
    top = max(c["eff"] for c in out)
    for c in out:
        c["type"] = "P" if c["eff"] == top else "E"
    return sorted(out, key=lambda c: c["cpu"])


def groups(top):
    """Listas de CPUs lógicas: P físicos (1 hilo por núcleo), hermanos SMT, E."""
    seen, p_first, p_sib, e = set(), [], [], []
    for c in top:
        if c["type"] == "E":
            e.append(c["cpu"])
        elif c["core"] in seen:
            p_sib.append(c["cpu"])
        else:
            seen.add(c["core"]); p_first.append(c["cpu"])
    return dict(P=p_first, P_smt=p_sib, E=e)


if __name__ == "__main__":
    t = topology()
    for c in t:
        print(f"CPU {c['cpu']:2d}  core {c['core']:2d}  eff {c['eff']}  {c['type']}")
    g = groups(t)
    print(g)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    with open(os.path.join(root, "results", "cpu_topology.json"), "w") as fh:
        json.dump(dict(cpus=t, groups=g), fh, indent=2)
