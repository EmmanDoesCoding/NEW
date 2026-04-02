"""
AVA_panda3d.py  -  Sign Language Avatar  (Panda3D  170 fps)
============================================================
pip install panda3d
python AVA_panda3d.py

SPACE = pause   R = restart   Q/ESC = quit
"""

import json, sys, os, numpy as np
from math import degrees, acos

from direct.showbase.ShowBase   import ShowBase
from direct.task                import Task
from direct.gui.OnscreenText    import OnscreenText
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import (
    Point3, Vec3, LColor, LVector3f, Quat,
    AmbientLight, DirectionalLight,
    WindowProperties, TextNode, ClockObject,
    NodePath,
)

# ── Config ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 900, 650
BG          = (0.12, 0.12, 0.15, 1)
SCALE       = 1.8

# ── Palette ───────────────────────────────────────────────────────────────────
def c(r,g,b): return LColor(r/255,g/255,b/255,1)
SKIN     = c(240, 180, 130)
SHIRT    = c(235, 235, 245)
SHIRT_HI = c(255, 255, 255)
EYE_W    = c(240,232,218)
IRIS     = c( 48, 30, 10)
PUPIL    = c(  6,  4,  2)
LIP      = c(108, 52, 42)
BROW     = c( 22, 12,  4)
WHITE    = LColor(1,1,1,1)

# ── Landmark indices ──────────────────────────────────────────────────────────
NOSE=0; LEYE=2; REYE=5; LEAR=7; REAR=8; ML=9; MR=10
LS=11; RS=12; LE=13; RE=14; LW=15; RW=16; LH=23; RH=24

HBONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
          (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
          (13,17),(17,18),(18,19),(19,20),(0,17)]

# ── Body centre calibration ───────────────────────────────────────────────────
# BCX / BCZ are the body centre offsets used by to3d().
# When running standalone (AVA_panda3d.py directly), we calibrate from
# clean_sign_motion.json if it exists, otherwise use MediaPipe defaults (0.5).
# When imported by main.py, calibrate() is called with real sign data.
BCX = 0.5   # default: MediaPipe horizontal centre
BCZ = 0.5   # default: MediaPipe vertical centre

def calibrate_from_file(json_file: str):
    """Calibrate BCX/BCZ from a JSON landmarks file."""
    global BCX, BCZ, data
    if not os.path.exists(json_file):
        print(f"[AVA] {json_file} not found — using default body centre (0.5, 0.5)")
        data = []
        return
    with open(json_file) as f:
        data = json.load(f)
    _xs, _zs = [], []
    for fr in data:
        if not fr.get("pose"): continue
        for i in range(min(17, len(fr["pose"]))):
            _xs.append(fr["pose"][i]["x"])
            _zs.append(fr["pose"][i]["y"])
    if _xs:
        BCX = (min(_xs)+max(_xs))/2
        BCZ = (min(_zs)+max(_zs))/2
    print(f"[AVA] Calibrated: BCX={BCX:.3f}  BCZ={BCZ:.3f}  ({len(data)} frames)")

def calibrate_from_frames(frames: list):
    """Calibrate BCX/BCZ from a list of frame dicts (called by main.py)."""
    global BCX, BCZ
    _xs, _zs = [], []
    for fr in frames:
        if not fr.get("pose"): continue
        for i in range(min(17, len(fr["pose"]))):
            _xs.append(fr["pose"][i]["x"])
            _zs.append(fr["pose"][i]["y"])
    if _xs:
        BCX = (min(_xs)+max(_xs))/2
        BCZ = (min(_zs)+max(_zs))/2

data = []   # standalone playback data (unused when imported by main.py)

# ── Coordinate helpers ────────────────────────────────────────────────────────
def to3d(lm):
    x =  (lm["x"] - BCX) * SCALE
    y =   lm["z"] * SCALE * 0.6
    z = -(lm["y"] - BCZ) * SCALE
    return Point3(x, y, z)

def flat(lm, y=0.0):
    return to3d(lm)

def flat3d(lm):
    return to3d(lm)


def pf(pose, idx):
    return to3d(pose[idx])

def sy(pt, y):
    """Return a copy of pt with Y replaced."""
    return Point3(pt.x, y, pt.z)


# ── Interpolation helpers ────────────────────────────────────────────────────
def lerp_lm(a, b, t):
    """Linearly interpolate between two landmark dicts."""
    return {
        "x": a["x"] + (b["x"] - a["x"]) * t,
        "y": a["y"] + (b["y"] - a["y"]) * t,
        "z": a["z"] + (b["z"] - a["z"]) * t,
    }

def lerp_frame(fa, fb, t):
    """Interpolate all landmarks between two frames."""
    out = {"pose": [], "left_hand": [], "right_hand": []}
    for key in ("pose", "left_hand", "right_hand"):
        la, lb = fa.get(key, []), fb.get(key, [])
        if la and lb and len(la) == len(lb):
            out[key] = [lerp_lm(a, b, t) for a, b in zip(la, lb)]
        else:
            out[key] = la or lb
    return out

# ── App ───────────────────────────────────────────────────────────────────────
class Avatar(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        props = WindowProperties()
        props.setTitle("Sign Language Avatar  |  Panda3D")
        props.setSize(WIN_W, WIN_H)
        self.win.requestProperties(props)
        self.setBackgroundColor(*BG)

        # ── Shader (one-time GPU compile — no per-frame cost) ────────────
        self.render.setShaderAuto()

        # ── Orbit camera — rotates around avatar centre ──────────────────
        self.disableMouse()   # disable default mouse control
        # Camera parent node sits at world origin — we rotate this to orbit
        self._cam_pivot = self.render.attachNewNode("cam_pivot")
        self.camera.reparentTo(self._cam_pivot)
        self.camera.setPos(0, -4.5, 0.3)   # camera offset from pivot
        self.camera.lookAt(0, 0, 0.2)

        # Orbit state
        self._orbit_h    = 0.0    # heading (left/right rotation)
        self._orbit_p    = -8.0   # pitch   (up/down tilt)
        self._orbit_zoom = 4.5    # distance from centre
        self._mouse_last = None
        self._orbit_btn  = False

        # Mouse button events
        self.accept("mouse1",           self._orbit_start)
        self.accept("mouse1-up",        self._orbit_stop)
        self.accept("wheel_up",         self._zoom_in)
        self.accept("wheel_down",       self._zoom_out)
        self.taskMgr.add(self._orbit_task, "orbit")

        self._lights()
        self.G = {}
        self._skeleton()

        # ── Ground shadow node (added after skeleton) ─────────────────────
        self._make_ground_shadow()

        self.hud = OnscreenText(
            text="", pos=(-1.28, 0.90), scale=0.046,
            fg=(0.75,0.75,0.75,1), align=TextNode.ALeft, mayChange=True)

        self.accept("space",  self._pause)
        self.accept("r",      self._restart)
        self.accept("q",      sys.exit)
        self.accept("escape", sys.exit)

        self.fi           = 0
        self.paused       = False
        self._anim_accum  = 0.0
        self._last_lh     = []
        self._last_rh     = []
        # Use first data frame if available, otherwise a safe empty frame
        _default = data[0] if data else {
            "pose": [{"x":0.5,"y":0.5,"z":-0.5}]*33,
            "left_hand":  [{"x":0.6,"y":0.88,"z":-0.65}]*21,
            "right_hand": [{"x":0.4,"y":0.88,"z":-0.65}]*21,
            "face": []
        }
        self._curr_frame  = _default
        self._prev_frame  = _default
        self.taskMgr.add(self._tick, "tick")

    # ── Orbit camera methods ──────────────────────────────────────────────────
    def _orbit_start(self):
        self._orbit_btn  = True
        self._mouse_last = None

    def _orbit_stop(self):
        self._orbit_btn  = False
        self._mouse_last = None

    def _zoom_in(self):
        self._orbit_zoom = max(2.0, self._orbit_zoom - 0.3)

    def _zoom_out(self):
        self._orbit_zoom = min(10.0, self._orbit_zoom + 0.3)

    def _orbit_task(self, task):
        if self._orbit_btn and self.mouseWatcherNode.hasMouse():
            mx = self.mouseWatcherNode.getMouseX()
            my = self.mouseWatcherNode.getMouseY()
            if self._mouse_last:
                dx = mx - self._mouse_last[0]
                dy = my - self._mouse_last[1]
                self._orbit_h -= dx * 120   # sensitivity
                self._orbit_p  = max(-80, min(80, self._orbit_p + dy * 80))
            self._mouse_last = (mx, my)

        # Apply rotation to pivot and zoom to camera
        self._cam_pivot.setHpr(self._orbit_h, self._orbit_p, 0)
        self.camera.setPos(0, -self._orbit_zoom, 0)
        self.camera.lookAt(self._cam_pivot)
        return Task.cont

    def _lights(self):
        for name, col, hpr in [
            ("key",   LColor(0.80, 0.76, 0.70, 1), (-30, -50,   0)),  # warm key — softer
            ("front", LColor(0.35, 0.36, 0.40, 1), (  0,   0,   0)),  # camera fill — softer
            ("fill",  LColor(0.18, 0.22, 0.38, 1), ( 80, -10,   0)),  # side fill
            ("rim",   LColor(0.40, 0.50, 0.70, 1), (180,  15,   0)),  # cool rim from behind
        ]:
            d = DirectionalLight(name); d.setColor(col)
            n = self.render.attachNewNode(d); n.setHpr(*hpr)
            self.render.setLight(n)
        a = AmbientLight("amb"); a.setColor(LColor(0.28, 0.28, 0.30, 1))
        self.render.setLight(self.render.attachNewNode(a))

    def _skeleton(self):
        def node(name, col):
            n = self.loader.loadModel("models/misc/sphere")
            n.reparentTo(self.render)
            n.setColor(col); n.hide()
            self.G[name] = n

        node("head", SKIN)
        for s in ("l","r"):
            node(f"ew_{s}", EYE_W); node(f"ei_{s}", IRIS)
            node(f"ep_{s}", PUPIL); node(f"es_{s}", WHITE)
            node(f"br_{s}", BROW)
        node("nose_t", SKIN); node("nose_b", SKIN)
        node("lip_l", LIP); node("lip_r", LIP); node("lip_m", LIP)
        node("neck", SKIN)

        for nm in ("t_l","t_r","t_top","t_bot","t_hi"):
            node(nm, SHIRT)   # all torso parts same dark shirt colour

        for s in ("l","r"):
            node(f"ua_{s}", SHIRT); node(f"la_{s}", SHIRT)
            node(f"sh_{s}", SHIRT); node(f"el_{s}", SHIRT)
            node(f"wr_{s}", SKIN)
            node(f"ua2_{s}", SHIRT)   # upper arm fill sphere
            node(f"la2_{s}", SHIRT)   # lower arm fill sphere

        for s in ("l","r"):
            for i in range(21):       node(f"hj_{s}_{i}", SKIN)
            for i in range(len(HBONES)): node(f"hb_{s}_{i}", SKIN)

    def _make_ground_shadow(self):
        """Soft dark ellipse on the ground below the avatar."""
        n = self.loader.loadModel("models/misc/sphere")
        n.reparentTo(self.render)
        n.setColor(LColor(0.06, 0.06, 0.08, 0.7))
        n.setPos(0, 0, -0.82)          # just below hip level
        n.setScale(0.22, 0.04, 0.14)   # small flat ellipse
        n.setTransparency(1)
        self.G["ground_shadow"] = n

    def _sph(self, name, pos, r):
        g = self.G[name]
        g.show(); g.setPos(pos); g.setScale(r,r,r); g.setHpr(0,0,0)

    def _lmb(self, name, p1, p2, r):
        g = self.G[name]
        d = p2 - p1; ln = d.length()
        if ln < 0.001: g.hide(); return
        g.show(); g.setPos((p1+p2)*0.5); g.setScale(r, r, ln*0.5)
        dn  = d/ln; zv = Vec3(0,0,1)
        dot = max(-1.0, min(1.0, zv.dot(Vec3(dn))))
        ang = degrees(acos(dot))
        ax  = zv.cross(Vec3(dn)); al = ax.length()
        if al > 1e-6:
            ax /= al; q = Quat()
            q.setFromAxisAngle(ang, LVector3f(ax.x,ax.y,ax.z))
            g.setQuat(self.render, q)
        else:
            g.setHpr(0, 180 if dn.z < 0 else 0, 0)

    def _tick(self, task):
        if self.paused: return Task.cont

        self._anim_accum += globalClock.getDt()
        INTERVAL = 1.0 / 30.0

        if self._anim_accum >= INTERVAL:
            self._anim_accum -= INTERVAL
            if not data:
                return Task.cont   # no data — main.py overrides this tick
            if self.fi >= len(data): self.fi = 0
            self._prev_frame = self._curr_frame
            self._curr_frame = data[self.fi]; self.fi += 1

        # Interpolation factor: how far between prev and curr frame
        t = min(1.0, self._anim_accum / max(1e-6, INTERVAL))

        # Blend between previous and current frame
        frame = lerp_frame(self._prev_frame, self._curr_frame, t)

        pose  = frame.get("pose", [])
        lh    = frame.get("left_hand",  [])
        rh    = frame.get("right_hand", [])
        # For bad fallback frames (z=0), use last good hand data
        if rh and abs(rh[0]["z"]) < 0.01:
            has_xy = any(abs(lm["x"]-0.5)>0.02 or abs(lm["y"]-0.5)>0.02
                         for lm in rh[:3])
            if not has_xy: rh = self._last_rh
        if lh and abs(lh[0]["z"]) < 0.01:
            has_xy = any(abs(lm["x"]-0.5)>0.02 or abs(lm["y"]-0.5)>0.02
                         for lm in lh[:3])
            if not has_xy: lh = self._last_lh
        if rh: self._last_rh = rh
        if lh: self._last_lh = lh
        if pose:
            self._pose(pose, lh, rh)

        self.hud.setText(
            f"Frame {self.fi}/{len(data)}  |  "
            f"render {globalClock.getAverageFrameRate():.0f}fps  anim 30fps\n"
            "SPACE=pause  R=restart  Q=quit")
        return Task.cont


    def _pose(self, pose, lh_lms, rh_lms):
        # Torso at Y=0 (background layer)
        ls  = pf(pose, LS);   rs  = pf(pose, RS)
        lhp = pf(pose, LH);   rhp = pf(pose, RH)
        sm  = (ls+rs)*0.5;    hm  = (lhp+rhp)*0.5

        # True 3D positions from pose data
        lec  = pf(pose, LE);  lwc = pf(pose, LW)
        rec  = pf(pose, RE);  rwc = pf(pose, RW)
        ls_a = ls;            rs_a = rs

        # ── Torso — solid filled with thick overlapping cylinders ────────
        self._lmb("t_l",   ls,  lhp, 0.092)   # left edge
        self._lmb("t_r",   rs,  rhp, 0.092)   # right edge
        self._lmb("t_top", ls,  rs,  0.088)   # shoulder bar
        self._lmb("t_bot", lhp, rhp, 0.080)   # hip bar
        self._lmb("t_hi",  sm,  hm,  0.090)   # centre fill (same depth, fills gap)

        # ── Arms (in front of torso) ───────────────────────────────────────
        self._lmb("ua_l",  ls_a, lec,  0.075)
        self._lmb("la_l",  lec,  lwc,  0.065)
        self._lmb("ua_r",  rs_a, rec,  0.075)
        self._lmb("la_r",  rec,  rwc,  0.065)
        self._sph("sh_l",  ls_a, 0.068)   # shoulder — slightly smaller than arm
        self._sph("sh_r",  rs_a, 0.068)
        self._sph("el_l",  lec,  0.058)   # elbow
        self._sph("el_r",  rec,  0.058)
        self._sph("wr_l",  lwc,  0.050)   # wrist
        self._sph("wr_r",  rwc,  0.050)
        # Fill spheres at mid-points of each segment to seal gaps
        self._sph("ua2_l", (ls_a+lec)*0.5, 0.065)
        self._sph("la2_l", (lec+lwc)*0.5,  0.058)
        self._sph("ua2_r", (rs_a+rec)*0.5, 0.065)
        self._sph("la2_r", (rec+rwc)*0.5,  0.058)

        # ── Head + neck ───────────────────────────────────────────────────
        hr = 0.21
        hc = Point3(sm.x, 0.0, sm.z + 0.40)
        nt = Point3(sm.x, 0.0, sm.z + 0.18)
        self._lmb("neck", sm, nt, 0.050)
        self._sph("head", hc, hr)
        self._face(pose, hc, hr)

        # ── Hands at Y=-0.22 (in front of arms) ──────────────────────────
        for side, lms, wpt in [("l", lh_lms, lwc), ("r", rh_lms, rwc)]:
            if not lms:
                for i in range(21):          self.G[f"hj_{side}_{i}"].hide()
                for i in range(len(HBONES)): self.G[f"hb_{side}_{i}"].hide()
                continue
            pts = [to3d(m) for m in lms]
            # Scale up finger spread from wrist
            HAND_SCALE = 1.4
            wp = pts[0]
            pts = [wp + (p - wp) * HAND_SCALE for p in pts]
            # Snap to arm wrist
            off = wpt - pts[0]
            pts = [p + off for p in pts]

            # ── Hand orientation from palm normal ─────────────────────────
            # Use wrist→index_mcp and wrist→pinky_mcp to compute palm facing
            if len(pts) > 17:
                v1 = pts[5]  - pts[0]   # wrist → index MCP
                v2 = pts[17] - pts[0]   # wrist → pinky MCP
                palm_n = v1.cross(v2)
                if palm_n.length() > 0.001:
                    palm_n.normalize()
                    # Rotate wrist cap to face palm direction
                    self.G[f"wr_{side}"].setQuat(
                        self.render,
                        self._vec_to_quat(palm_n))

            for i, p in enumerate(pts):
                self._sph(f"hj_{side}_{i}", p, 0.030)
            for i, (a, b) in enumerate(HBONES):
                if a < len(pts) and b < len(pts):
                    self._lmb(f"hb_{side}_{i}", pts[a], pts[b], 0.022)
                else:
                    self.G[f"hb_{side}_{i}"].hide()

    def _vec_to_quat(self, v):
        """Return a Quat that rotates Z-axis to align with vector v."""
        from panda3d.core import Vec3 as V3
        zv  = V3(0,0,1)
        vn  = V3(v.x, v.y, v.z)
        vn.normalize()
        dot = max(-1.0, min(1.0, zv.dot(vn)))
        ang = degrees(acos(dot))
        ax  = zv.cross(vn)
        al  = ax.length()
        q   = Quat()
        if al > 1e-6:
            ax /= al
            q.setFromAxisAngle(ang, LVector3f(ax.x, ax.y, ax.z))
        return q

    def _face(self, pose, hc, hr):
        ey = hr * 0.20
        FY = -hr * 0.90   # face protrudes toward camera (negative Y)

        for sx in (-1, 1):
            s  = "l" if sx == -1 else "r"
            ep = Point3(hc.x + sx*hr*0.34, hc.y + FY, hc.z + hr*0.12)
            self._sph(f"ew_{s}", ep,                                    ey)
            self._sph(f"ei_{s}", Point3(ep.x, ep.y-0.015, ep.z),       ey*0.68)
            self._sph(f"ep_{s}", Point3(ep.x, ep.y-0.028, ep.z),       ey*0.36)
            self._sph(f"es_{s}", Point3(ep.x+ey*0.18, ep.y-0.036,
                                         ep.z+ey*0.18),                 ey*0.12)

        ear_dy = (pose[LEAR]["y"] - pose[REAR]["y"]) * hr * 1.2
        for sx in (-1, 1):
            s    = "l" if sx == -1 else "r"
            tilt = sx * ear_dy * 0.3
            bc   = Point3(hc.x + sx*hr*0.34, hc.y + FY + 0.01,
                          hc.z + hr*0.33 + tilt)
            b1   = Point3(bc.x - hr*0.15, bc.y, bc.z - tilt*0.5)
            b2   = Point3(bc.x + hr*0.15, bc.y, bc.z + tilt*0.5)
            self._lmb(f"br_{s}", b1, b2, hr*0.030)

        ntip = Point3(hc.x, hc.y + FY - 0.02, hc.z - hr*0.10)
        ntop = Point3(hc.x, hc.y + FY + 0.01, hc.z + hr*0.12)
        self._lmb("nose_b", ntop, ntip, hr*0.038)
        self._sph("nose_t", ntip,        hr*0.058)

        ml_lm   = pose[ML]; mr_lm = pose[MR]; le_lm = pose[LEYE]
        ear_sep = max(0.001, abs(pose[LEAR]["x"] - pose[REAR]["x"]))
        mw = float(np.clip(
            abs(ml_lm["x"]-mr_lm["x"]) / ear_sep * hr * 0.85,
            hr*0.13, hr*0.33))
        face_h = abs(ml_lm["y"] - le_lm["y"])
        mz = float(-np.clip(face_h * 2.6 * hr, hr*0.20, hr*0.40))
        MY = hc.y + FY - 0.01
        ml = Point3(hc.x - mw, MY, hc.z + mz)
        mr = Point3(hc.x + mw, MY, hc.z + mz)
        self._sph("lip_l", ml,     hr*0.032)
        self._sph("lip_r", mr,     hr*0.032)
        self._lmb("lip_m", ml, mr, hr*0.028)

    def _pause(self):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")

    def _restart(self):
        self.fi = 0; print("Restarted")

if __name__ == "__main__":
    calibrate_from_file("clean_sign_motion.json")
    if not data:
        print("No data to play. Put clean_sign_motion.json next to this file.")
        print("Running with empty animation (rest pose only).")
    Avatar().run()