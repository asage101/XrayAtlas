import cv2

def get_region_map():
    width, height = 1252, 2448

    manual_region_map = {
        'Skull': (
            int(width * 0.39),
            0,
            int(width * 0.22),
            int(height * 0.15)
        ),
        'Sinus': (
            int(width * 0.43),
            int(height * 0.07),
            int(width * 0.14),
            int(height * 0.03)
        ),
        'Cervical Spine': (
            int(width * 0.41),
            int(height * 0.12),
            int(width * 0.20),
            int(height * 0.03)
        ),

        'Clavicles': (
            int(width * 0.17),
            int(height * 0.14),
            int(width * 0.65),
            int(height * 0.13)
        ),
        'Shoulder': (
            int(width * 0.17),
            int(height * 0.12),
            int(width * 0.20),
            int(height * 0.18)
        ),
        'Chest': (
            int(width * 0.30),
            int(height * 0.12),
            int(width * 0.39),
            int(height * 0.26)
        ),
        'Thoracic Spine': (
            int(width * 0.43),
            int(height * 0.14),
            int(width * 0.14),
            int(height * 0.20)
        ),

        'Abdomen': (
            int(width * 0.33),
            int(height * 0.26),
            int(width * 0.33),
            int(height * 0.23)
        ),
        'Pelvis': (
            int(width * 0.33),
            int(height * 0.37),
            int(width * 0.33),
            int(height * 0.15)
        ),
        'Hip': (
            int(width * 0.29),
            int(height * 0.38),
            int(width * 0.23),
            int(height * 0.17)
        ),
        'Lumbar Spine': (
            int(width * 0.43),
            int(height * 0.31),
            int(width * 0.14),
            int(height * 0.10)
        ),

        'Thigh': (
            int(width * 0.29),
            int(height * 0.51),
            int(width * 0.23),
            int(height * 0.20)
        ),
        'Knee': (
            int(width * 0.29),
            int(height * 0.68),
            int(width * 0.23),
            int(height * 0.07)
        ),
        'Lower Leg': (
            int(width * 0.29),
            int(height * 0.72),
            int(width * 0.23),
            int(height * 0.20)
        ),
        'Ankle': (
            int(width * 0.29),
            int(height * 0.8),
            int(width * 0.23),
            int(height * 0.14)
        ),
        'Feet': (
            int(width * 0.29),
            int(height * 0.93),
            int(width * 0.23),
            int(height * 0.07)
        ),

        'Elbow': (
            int(width * 0.17),
            int(height * 0.3),
            int(width * 0.15),
            int(height * 0.1)
        ),
        'Forearm': (
            int(width * 0.12),
            int(height * 0.35),
            int(width * 0.15),
            int(height * 0.13)
        ),
        'Wrist': (
            int(width * 0.11),
            int(height * 0.46),
            int(width * 0.15),
            int(height * 0.05)
        ),
        'Hand': (
            int(width * 0.05),
            int(height * 0.47),
            int(width * 0.17),
            int(height * 0.12)
        ),
        'Finger': (
            int(width * 0.06),
            int(height * 0.5),
            int(width * 0.15),
            int(height * 0.08)
        )
    }
    return manual_region_map


def extract_regions(template_path, manual_region_map):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("Error: Cannot load template image from", template_path)
        return {}

    regions = {}
    for label, bbox in manual_region_map.items():
        if bbox is None:
            continue
        x, y, w, h = bbox
        crop = template[y:y + h, x:x + w].copy()
        regions[label] = crop
    return regions
