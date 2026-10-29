import cv2
import numpy as np
import argparse

CFG = {
    "basket": dict(
        lower=(4, 90, 60), upper=(20, 255, 255),
        min_area=40, min_circ=0.45, asp=None, min_fill=None,
        max_jump=250, proc_w=None, blur=9, morph="old",
        # son konumun yakınında kullanılan gevşek ayar
        loose=dict(lower=(2, 60, 35), upper=(22, 255, 255),
                   min_area=25, min_circ=0.25),
    ),
    "tenis": dict(
        lower=(25, 80, 90), upper=(42, 255, 255),
        min_area=250, min_circ=0.60, asp=(0.6, 1.6), min_fill=0.55,
        max_jump=150, proc_w=640, blur=7, morph="new",
        loose=None,
    ),
}

DEADZONE = 40
PRINT_INTERVAL = 0.1   # 100 ms
HOLD_FRAMES = 10       # top kaybolunca tahmini konumu kaç kare göstersin
SEARCH_LOST = 12       # kaç kare boyunca son konum çevresinde aransın
CUT_THRESHOLD = 40
EDGE = 3               # px, kareye değme toleransı


def make_mask(img, lower, upper, blur, morph):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (blur, blur), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    if morph == "old":
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
    else:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def pick(mask, p, ref, max_dist, shape, strict_shape=True):
    Hh, Ww = shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if area < p["min_area"] or per == 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        touches_edge = (x <= EDGE or y <= EDGE or
                        x + w >= Ww - EDGE or y + h >= Hh - EDGE)
        cx, cy = x + w // 2, y + h // 2

        # kareyi kesen top (özellikle üst kenar): şekil kontrolü yapma
        skip_shape = touches_edge and ref is not None
        if not skip_shape:
            circ = 4 * np.pi * area / per ** 2
            if circ < p["min_circ"]:
                continue
            if strict_shape:
                if p.get("asp") is not None and not (p["asp"][0] <= w / h <= p["asp"][1]):
                    continue
                if p.get("min_fill") is not None:
                    (_, _), r = cv2.minEnclosingCircle(c)
                    if area / (np.pi * r * r + 1e-6) < p["min_fill"]:
                        continue
        else:
            circ = 0.5

        dist = 0.0
        if ref is not None:
            dist = np.hypot(cx - ref[0], cy - ref[1])
            if dist > max_dist:
                continue
        score = area * max(circ, 0.2) / (1 + dist / 60.0)
        if score > best_score:
            best, best_score = (cx, cy, x, y, w, h), score
    return best


def detect(img, cfg, ref, lost):
    mask = make_mask(img, cfg["lower"], cfg["upper"], cfg["blur"], cfg["morph"])
    max_dist = min(cfg["max_jump"] + 15 * lost, 400)
    det = pick(mask, cfg, ref, max_dist, img.shape)
    tracked_loose = False
    # sıkı filtre bulamadıysa: son konum çevresinde gevşek ara
    if det is None and ref is not None and cfg["loose"] is not None:
        lo = cfg["loose"]
        lmask = make_mask(img, lo["lower"], lo["upper"], cfg["blur"], cfg["morph"])
        det = pick(lmask, lo, ref, max_dist * 0.8, img.shape, strict_shape=False)
        if det is not None:
            tracked_loose = True
            mask = lmask
    return det, mask, tracked_loose


def resize_for(frame, cfg):
    pw = cfg["proc_w"]
    if pw and frame.shape[1] > pw:
        s = frame.shape[1] / pw
        return cv2.resize(frame, None, fx=1 / s, fy=1 / s), s
    return frame, 1.0


def auto_select(cap):
    counts = {k: 0 for k in CFG}
    for _ in range(45):
        ok, frame = cap.read()
        if not ok:
            break
        for name, cfg in CFG.items():
            img, _ = resize_for(frame, cfg)
            if detect(img, cfg, None, 99)[0] is not None:
                counts[name] += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return max(counts, key=counts.get)


def maneuver(dx):
    if abs(dx) <= DEADZONE:
        return "MERKEZDE"
    return "SOLA DON" if dx < 0 else "SAGA DON"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ball", choices=["auto", "basket", "tenis"], default="auto")
    ap.add_argument("--save", help="cikti video yolu")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Video acilamadi")
        return

    ball = auto_select(cap) if args.ball == "auto" else args.ball
    cfg = CFG[ball]
    print(f"Takip edilen top: {ball}  (q: cik, m: maske)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center = (W // 2, H // 2)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    last_pos, prev_pos, last_det = None, None, None   # işleme boyutunda konumlar
    lost, last_print, idx = 99, -1.0, 0
    prev_gray, show_mask = None, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / fps

        img, scale = resize_for(frame, cfg)

        # sahne kesmesi: hafızayı sıfırla
        gray = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and cv2.absdiff(gray, prev_gray).mean() > CUT_THRESHOLD:
            last_pos, prev_pos, last_det, lost = None, None, None, 99
        prev_gray = gray

        # arama merkezi: son konum + hız * kayıp kare sayısı
        ref = None
        if last_pos is not None and lost <= SEARCH_LOST:
            if prev_pos is not None and lost > 0:
                vx = np.clip(last_pos[0] - prev_pos[0], -40, 40)
                vy = np.clip(last_pos[1] - prev_pos[1], -40, 40)
                ref = (last_pos[0] + vx * lost, last_pos[1] + vy * lost)
            else:
                ref = last_pos

        det, mask, loose_used = detect(img, cfg, ref, lost)

        if det is not None:
            lost = 0
            prev_pos, last_pos = last_pos, (det[0], det[1])
            last_det = tuple(int(v * scale) for v in det)
        else:
            lost += 1

        cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.line(frame, (center[0] - DEADZONE, 0), (center[0] - DEADZONE, H), (150, 150, 150), 1)
        cv2.line(frame, (center[0] + DEADZONE, 0), (center[0] + DEADZONE, H), (150, 150, 150), 1)

        text = "TOP YOK"
        if last_det is not None and lost <= HOLD_FRAMES:
            cx, cy, x, y, w, h = last_det
            dx, dy = cx - center[0], cy - center[1]
            text = maneuver(dx)

            predicted = lost > 0
            box_color = (0, 255, 255) if predicted else (0, 255, 0)  # sarı: tahmin
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            length = max(np.hypot(dx, dy), 1)
            cv2.arrowedLine(frame, center, (cx, cy), (0, 0, 255), 4,
                            tipLength=min(0.5, 30.0 / length))
            cv2.putText(frame, f"({cx},{cy}) dx={dx} dy={dy}", (x, max(y - 12, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            if t - last_print >= PRINT_INTERVAL:
                tag = " (tahmin)" if predicted else ""
                print(f"kare={idx:4d} t={t:6.2f}s  merkez=({cx}, {cy})  dx={dx}  -> {text}{tag}")
                last_print = t
        elif t - last_print >= PRINT_INTERVAL:
            print(f"kare={idx:4d} t={t:6.2f}s  top bulunamadi")
            last_print = t

        cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        cv2.putText(frame, f"kare {idx}", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow("Takip", frame)
        if show_mask:
            cv2.imshow("Maske", mask)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Maske")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
