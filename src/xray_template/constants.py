SITE_LABELS = [
    'Abdomen',
    'Ankle',
    'Cervical Spine',
    'Chest',
    'Clavicles',
    'Elbow',
    'Feet',
    'Finger',
    'Forearm',
    'Hand',
    'Hip',
    'Knee',
    'Lower Leg',
    'Lumbar Spine',
    'Pelvis',
    'Shoulder',
    'Sinus',
    'Skull',
    'Thigh',
    'Thoracic Spine',
    'Wrist'
]

DUPLICATE_SITES = ['Hip', 'Thigh', 'Lower Leg']

MIRROR_MAP = {
    # arms
    'Elbow': 'Elbow_right',
    # 'Elbow Forearm': 'Elbow Forearm_right',
    'Forearm': 'Forearm_right',
    'Hand': 'Hand_right',
    # 'Finger': 'Finger_right',
    # 'Finger Hand': 'Finger Hand_right',
    # 'Wrist': 'Wrist_right',
    'Shoulder': 'Shoulder_right',
    # 'Forearm Shoulder': 'Forearm Shoulder_right',

    # legs
    'Hip': 'Hip_right',
    # 'Hip Thigh': 'Hip Thigh_right',
    # 'Hip Knee Thigh': 'Hip Knee Thigh_right',
    'Thigh': 'Thigh_right',
    'Knee': 'Knee_right',
    # 'Knee Lower Leg': 'Knee Lower Leg_right',
    'Lower Leg': 'Lower Leg_right',
    'Ankle': 'Ankle_right',
    # 'Ankle Feet': 'Ankle Feet_right',
    # 'Ankle Knee Lower Leg': 'Ankle Knee Lower Leg_right',
    'Feet': 'Feet_right'
}

BASE_POSITIONS = {
    'Skull': (500, 80),
    # 'Sinus': (500, 160),
    'Cervical Spine': (500, 208),
    # 'Thoracic Spine': (500, 300),
    'Clavicles': (500, 336),
    'Chest': (500, 464),
    'Abdomen': (500, 592),
    # 'Lumbar Spine': (500, 540),
    'Pelvis': (500, 720),
    'Hip': (440, 848),
    'Thigh': (430, 976),
    'Knee': (420, 1104),
    'Lower Leg': (410, 1232),
    'Ankle': (400, 1360),
    'Feet': (380, 1488),

    # Left arm
    'Shoulder': (372, 336),
    'Elbow': (340, 464),
    'Forearm': (308, 592),
    # 'Wrist': (286, 440),
    'Hand': (244, 720),
    # 'Finger': (244, 848),

    # Misc
    # 'Others': (100, 100)
}



# Mirror right side (flipped horizontally around canvas center ~500)
SITE_POSITIONS_RIGHT = {
    f"{label}_right": (1000 - x, y)
    for label, (x, y) in BASE_POSITIONS.items()
    if label in MIRROR_MAP
}

SITE_POSITIONS = {}

# Assign each SITE_LABEL to an anchor position based on keyword
for label in SITE_LABELS:
    if label.endswith("_right"):
        continue

    for anchor, (x, y) in BASE_POSITIONS.items():
        if anchor in label:
            SITE_POSITIONS[label] = (x, y)
            break
    else:
        SITE_POSITIONS[label] = (50, 1900)  # fallback

# Add mirrored positions
SITE_POSITIONS.update(SITE_POSITIONS_RIGHT)

# Canvas size
CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 1800
CANVAS_SIZE = (CANVAS_HEIGHT, CANVAS_WIDTH)