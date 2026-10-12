# orbdetpy_guard.py
# -----------------------------------------------------------------------------
# orbdetpy runs a Java server in the background. A new script that starts while
# the previous script's server is still shutting down can attach to that dying
# server, then waits 30 s and crashes with grpc.FutureTimeoutError (see v3.md
# Section 19.6). clear_orbdetpy_servers() kills any leftover orbdetpy server
# (only processes running orbdetpy's own .jar, never other Java programs) and
# waits until none remain. Call it between any two scripts that use orbdetpy.
# Run directly:  python orbdetpy_guard.py
# -----------------------------------------------------------------------------
import time, psutil
from os import path
from orbdetpy import _jar_file
def clear_orbdetpy_servers(timeout=15.0):
    """Kill any leftover orbdetpy Java server (only processes running orbdetpy's own jar), wait until gone."""
    jar = path.split(_jar_file)[-1]
    def servers():
        out = []
        for p in psutil.process_iter(attrs=["cmdline"]):
            cl = p.info.get("cmdline")
            if cl and cl[0].endswith("java") and any(x.endswith(jar) for x in cl):
                out.append(p)
        return out
    t0 = time.time()
    while servers() and time.time() - t0 < timeout:
        for p in servers():
            try: p.kill()
            except psutil.Error: pass
        time.sleep(0.2)
    return not servers()
if __name__ == "__main__":
    print("cleared" if clear_orbdetpy_servers() else "STILL RUNNING")