import ast
import json
import operator
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ============================================================
# AYARLAR
# ============================================================

MODEL_PATH = "math_symbols_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"

CONFIDENCE_THRESHOLD = 0.70

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 500

BRUSH_SIZE = 14
ERASER_SIZE = 45

RECOGNITION_DELAY = 0.30

WINDOW_NAME = "CNN Matematik Tahtasi"


# ============================================================
# MODELİ VE SINIFLARI YÜKLE
# ============================================================

model = load_model(MODEL_PATH)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)

print("Model yuklendi.")
print("Siniflar:", class_names)
print("Model girisi:", model.input_shape)
print("Model cikisi:", model.output_shape)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "Model cikis sayisi ile class_names.json "
        "sinif sayisi ayni degil."
    )


# Roboflow sınıf isimlerini matematik sembollerine çevir
symbol_map = {
    "div": "/",
    "eqv": "=",
    "minus": "-",
    "mult": "*",
    "plus": "+"
}


# ============================================================
# GÜVENLİ MATEMATİK HESAPLAMA
# ============================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}


def evaluate_expression(expression):
    """
    Yalnızca sayılar ve + - * / işlemlerine izin verir.
    """

    def evaluate_node(node):
        if isinstance(node, ast.Expression):
            return evaluate_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Gecersiz sabit")

        if isinstance(node, ast.BinOp):
            operation_type = type(node.op)

            if operation_type not in allowed_operators:
                raise ValueError("Desteklenmeyen operator")

            left_value = evaluate_node(node.left)
            right_value = evaluate_node(node.right)

            return allowed_operators[operation_type](
                left_value,
                right_value
            )

        raise ValueError("Gecersiz ifade")

    expression_tree = ast.parse(
        expression,
        mode="eval"
    )

    return evaluate_node(expression_tree)


def format_result(result):
    """
    5.0 değerini 5, uzun ondalıkları daha kısa gösterir.
    """

    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))

        return f"{result:.6f}".rstrip("0").rstrip(".")

    return str(result)


# ============================================================
# SEMBOLÜ CNN GİRİŞİNE DÖNÜŞTÜR
# ============================================================

def prepare_symbol(symbol_image):
    """
    İkili sembol görüntüsünü:
        28x28
        siyah arka plan
        beyaz sembol
        0-1 normalizasyon

    biçimine dönüştürür.
    """

    points = cv2.findNonZero(symbol_image)

    if points is None:
        return None, None

    x, y, w, h = cv2.boundingRect(points)

    symbol = symbol_image[
        y:y + h,
        x:x + w
    ]

    if symbol.size == 0:
        return None, None

    target_size = 20

    if h > w:
        new_h = target_size
        new_w = max(
            1,
            int(w * target_size / h)
        )
    else:
        new_w = target_size
        new_h = max(
            1,
            int(h * target_size / w)
        )

    resized = cv2.resize(
        symbol,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    prepared_image = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    prepared_image[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    model_input = (
        prepared_image.astype("float32") / 255.0
    )

    model_input = model_input.reshape(
        28,
        28,
        1
    )

    return model_input, prepared_image


# ============================================================
# PARÇALI SEMBOLLERİ BİRLEŞTİR
# ============================================================

def horizontal_overlap(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    overlap_start = max(ax, bx)
    overlap_end = min(ax + aw, bx + bw)

    overlap_width = max(
        0,
        overlap_end - overlap_start
    )

    return overlap_width / max(
        1,
        min(aw, bw)
    )


def should_merge_boxes(box_a, box_b):
    """
    ÷, = ve kopuk çizilmiş sembollerin parçalarının
    aynı karakter olup olmadığını belirler.
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    overlap_ratio = horizontal_overlap(
        box_a,
        box_b
    )

    center_a_x = ax + aw / 2
    center_b_x = bx + bw / 2

    center_distance_x = abs(
        center_a_x - center_b_x
    )

    close_on_x = center_distance_x < (
        max(aw, bw) * 0.42
    )

    return (
        overlap_ratio > 0.45
        or close_on_x
    )


def merge_two_boxes(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = min(ax, bx)
    y1 = min(ay, by)

    x2 = max(
        ax + aw,
        bx + bw
    )

    y2 = max(
        ay + ah,
        by + bh
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


def merge_component_boxes(boxes):
    """
    Aynı X bölgesindeki ayrı parçaları birleştirir.

    Örnek:
        ÷ → üst nokta + çizgi + alt nokta
        = → üst çizgi + alt çizgi
    """

    boxes = boxes.copy()

    changed = True

    while changed:
        changed = False

        merged_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            current_box = boxes[i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                if should_merge_boxes(
                    current_box,
                    boxes[j]
                ):
                    current_box = merge_two_boxes(
                        current_box,
                        boxes[j]
                    )

                    used[j] = True
                    changed = True

            merged_boxes.append(current_box)

        boxes = merged_boxes

    return boxes


# ============================================================
# ÇİZİMDEKİ BÜTÜN SEMBOLLERİ BUL
# ============================================================

def find_symbols(input_canvas):
    gray = cv2.cvtColor(
        input_canvas,
        cv2.COLOR_BGR2GRAY
    )

    # Siyah çizgileri beyaz, arka planı siyah yap
    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Çok küçük kopuklukları birleştir
    kernel = np.ones(
        (2, 2),
        dtype=np.uint8
    )

    binary = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    (
        component_count,
        component_labels,
        component_stats,
        component_centers
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    canvas_height, canvas_width = binary.shape

    boxes = []

    # Sıfırıncı component arka plandır
    for component_id in range(1, component_count):
        x = component_stats[
            component_id,
            cv2.CC_STAT_LEFT
        ]

        y = component_stats[
            component_id,
            cv2.CC_STAT_TOP
        ]

        w = component_stats[
            component_id,
            cv2.CC_STAT_WIDTH
        ]

        h = component_stats[
            component_id,
            cv2.CC_STAT_HEIGHT
        ]

        area = component_stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        # Küçük gürültüleri ele
        if area < 25:
            continue

        if w < 3 or h < 3:
            continue

        # Kenara değen anormal bölgeleri ele
        if (
            x <= 2
            or y <= 2
            or x + w >= canvas_width - 2
            or y + h >= canvas_height - 2
        ):
            continue

        boxes.append(
            (x, y, w, h)
        )

    boxes = merge_component_boxes(boxes)

    boxes = [
        box
        for box in boxes
        if box[2] >= 5
        and box[3] >= 8
    ]

    # Karakterleri soldan sağa sırala
    boxes.sort(
        key=lambda box: box[0]
    )

    return binary, boxes


# ============================================================
# BÜTÜN SEMBOLLERİ TEK BATCH İÇİNDE TANI
# ============================================================

def recognize_symbols(binary, boxes):
    prepared_samples = []
    valid_items = []

    for x, y, w, h in boxes:
        padding = 6

        crop_x1 = max(
            0,
            x - padding
        )

        crop_y1 = max(
            0,
            y - padding
        )

        crop_x2 = min(
            binary.shape[1],
            x + w + padding
        )

        crop_y2 = min(
            binary.shape[0],
            y + h + padding
        )

        symbol_crop = binary[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        model_input, prepared_image = prepare_symbol(
            symbol_crop
        )

        if model_input is None:
            continue

        prepared_samples.append(
            model_input
        )

        valid_items.append({
            "box": (x, y, w, h),
            "prepared": prepared_image
        })

    if not prepared_samples:
        return []

    batch = np.array(
        prepared_samples,
        dtype=np.float32
    )

    # Bütün karakterleri tek seferde CNN'e ver
    all_probabilities = model(
        batch,
        training=False
    ).numpy()

    detections = []

    for item, probabilities in zip(
        valid_items,
        all_probabilities
    ):
        class_id = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_id]
        )

        class_name = class_names[
            class_id
        ]

        displayed_symbol = symbol_map.get(
            class_name,
            class_name
        )

        detections.append({
            "symbol": displayed_symbol,
            "class_name": class_name,
            "confidence": confidence,
            "box": item["box"],
            "prepared": item["prepared"]
        })

    return detections


# ============================================================
# ALGILANAN SEMBOLLERDEN İFADE OLUŞTUR
# ============================================================

def build_expression(detections):
    tokens = []

    for detection in detections:
        confidence = detection[
            "confidence"
        ]

        if confidence < CONFIDENCE_THRESHOLD:
            tokens.append("?")
        else:
            tokens.append(
                detection["symbol"]
            )

    visible_expression = "".join(
        tokens
    )

    # Eşittirin sol tarafını hesapla
    calculation_expression = (
        visible_expression
        .split("=")[0]
    )

    return (
        visible_expression,
        calculation_expression
    )


# ============================================================
# ÇİZİM TAHTASINI OLUŞTUR
# ============================================================

canvas = np.full(
    (
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        3
    ),
    255,
    dtype=np.uint8
)

drawing = False
erasing = False

last_point = None
last_draw_time = time.time()

needs_recognition = True

cached_binary = np.zeros(
    (
        CANVAS_HEIGHT,
        CANVAS_WIDTH
    ),
    dtype=np.uint8
)

cached_detections = []
cached_visible_expression = ""
cached_result = None


# ============================================================
# FARE OLAYLARI
# ============================================================

def mouse_callback(event, x, y, flags, parameter):
    global drawing
    global erasing
    global last_point
    global last_draw_time
    global needs_recognition

    # Sol fare tuşuna basıldı
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        erasing = False

        last_point = (x, y)

        cv2.circle(
            canvas,
            (x, y),
            BRUSH_SIZE // 2,
            (0, 0, 0),
            -1,
            lineType=cv2.LINE_AA
        )

        last_draw_time = time.time()
        needs_recognition = True

    # Sağ fare tuşuna basıldı
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        erasing = True

        last_point = (x, y)

        cv2.circle(
            canvas,
            (x, y),
            ERASER_SIZE // 2,
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA
        )

        last_draw_time = time.time()
        needs_recognition = True

    # Fare hareket ediyor
    elif (
        event == cv2.EVENT_MOUSEMOVE
        and drawing
    ):
        if erasing:
            color = (255, 255, 255)
            thickness = ERASER_SIZE
        else:
            color = (0, 0, 0)
            thickness = BRUSH_SIZE

        cv2.line(
            canvas,
            last_point,
            (x, y),
            color,
            thickness,
            lineType=cv2.LINE_AA
        )

        last_point = (x, y)
        last_draw_time = time.time()
        needs_recognition = True

    # Fare tuşları bırakıldı
    elif event in (
        cv2.EVENT_LBUTTONUP,
        cv2.EVENT_RBUTTONUP
    ):
        drawing = False
        erasing = False
        last_point = None

        last_draw_time = time.time()
        needs_recognition = True


# ============================================================
# PENCEREYİ HAZIRLA
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    CANVAS_WIDTH,
    CANVAS_HEIGHT
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# Modeli bir kez boş veriyle çalıştırarak hazırla
warmup_input = np.zeros(
    (1, 28, 28, 1),
    dtype=np.float32
)

model(
    warmup_input,
    training=False
)


# ============================================================
# ANA PROGRAM DÖNGÜSÜ
# ============================================================

while True:
    recognition_ready = (
        needs_recognition
        and not drawing
        and (
            time.time() - last_draw_time
            > RECOGNITION_DELAY
        )
    )

    if recognition_ready:
        (
            cached_binary,
            boxes
        ) = find_symbols(canvas)

        cached_detections = recognize_symbols(
            cached_binary,
            boxes
        )

        (
            cached_visible_expression,
            calculation_expression
        ) = build_expression(
            cached_detections
        )

        cached_result = None

        # Cevap yalnızca eşittir yazılmışsa gösterilir
        if (
            "=" in cached_visible_expression
            and calculation_expression
            and "?" not in calculation_expression
        ):
            try:
                result = evaluate_expression(
                    calculation_expression
                )

                cached_result = format_result(
                    result
                )

            except ZeroDivisionError:
                cached_result = "Sifira bolunemez"

            except Exception:
                cached_result = None

        needs_recognition = False

    # Canvas'ın kopyasına kutu ve cevap ekle
    display = canvas.copy()

    # Algılanan karakter kutularını çiz
    for detection in cached_detections:
        x, y, w, h = detection["box"]

        confidence = detection[
            "confidence"
        ]

        if confidence >= CONFIDENCE_THRESHOLD:
            color = (0, 180, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        label = (
            f"{detection['symbol']} "
            f"%{confidence * 100:.0f}"
        )

        label_y = max(
            65,
            y - 10
        )

        cv2.putText(
            display,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    # Eşittir işaretinin sağına cevabı yaz
    if cached_result is not None:
        equals_detections = [
            detection
            for detection in cached_detections
            if (
                detection["symbol"] == "="
                and detection["confidence"]
                >= CONFIDENCE_THRESHOLD
            )
        ]

        if equals_detections:
            equals_detection = (
                equals_detections[-1]
            )

            eq_x, eq_y, eq_w, eq_h = (
                equals_detection["box"]
            )

            answer_x = eq_x + eq_w + 30
            answer_y = eq_y + eq_h

            cv2.putText(
                display,
                cached_result,
                (answer_x, answer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 80, 0),
                5,
                cv2.LINE_AA
            )

    # Üst bilgi paneli
    cv2.rectangle(
        display,
        (0, 0),
        (CANVAS_WIDTH, 50),
        (235, 235, 235),
        -1
    )

    expression_text = (
        f"Algilanan: "
        f"{cached_visible_expression}"
    )

    cv2.putText(
        display,
        expression_text,
        (15, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    controls_text = (
        "Sol: Ciz | Sag: Sil | "
        "C: Temizle | S: Kaydet | Q: Cikis"
    )

    cv2.putText(
        display,
        controls_text,
        (600, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    key = cv2.waitKey(10) & 0xFF

    # Q: Programdan çık
    if key == ord("q"):
        break

    # C: Çizim tahtasını temizle
    elif key == ord("c"):
        canvas[:] = 255

        cached_binary[:] = 0
        cached_detections = []
        cached_visible_expression = ""
        cached_result = None

        needs_recognition = True
        last_draw_time = time.time()

    # S: Sonuç görüntüsünü kaydet
    elif key == ord("s"):
        cv2.imwrite(
            "hesaplama_sonucu.png",
            display
        )

        print(
            "hesaplama_sonucu.png kaydedildi."
        )


cv2.destroyAllWindows()