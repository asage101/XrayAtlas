import numpy as np
import cv2
from .utils import display_image
import SimpleITK as sitk

def blur_then_sharpen_image(img):
    # Define the kernel for sharpening (a common 3x3 sharpening filter)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    
    # Apply convolution using the sharpening kernel
    sharpened = cv2.filter2D(img, -1, kernel)
    
    return sharpened

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert to grayscale if needed
    if img.dtype == np.float32 or img.dtype == np.float64:
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = norm_img.astype(np.uint8)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    # Convert back to BGR if original was color
    if len(img.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        return enhanced
    
def rotate_image(image, angle_deg, scale=1, fill_value=255):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    # rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
    #                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def create_foreground_mask_white_bg(image, white_thresh=240):
    mask = (image < white_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def create_foreground_mask_white_bg_f32(image, white_thresh):
    mask = (image < white_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def compute_bounding_box(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def compute_principal_angle(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 2:
        return 0.0
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eig(cov)
    principal_vec = eigenvecs[:, np.argmax(eigenvals)]
    angle = np.arctan2(principal_vec[1], principal_vec[0])
    angle_deg = np.degrees(angle)
    if abs(angle_deg) > 45:
        if angle_deg > 0:
            angle_deg -= 90
        else:
            angle_deg += 90
    return angle_deg


def deskew_image_by_mask(image, mask):
    angle_deg = compute_principal_angle(mask)
    rotated_img = rotate_image(image, -angle_deg, fill_value=255)
    rotated_mask = rotate_image(mask * 255, -angle_deg, fill_value=0)
    rotated_mask = (rotated_mask > 127).astype(np.uint8)
    return rotated_img, rotated_mask


def overlay_for_region(new_xray, region, deskew=False, white_thresh=240):
    mask = create_foreground_mask_white_bg(new_xray, white_thresh)
    if deskew:
        new_xray, mask = deskew_image_by_mask(new_xray, mask)

    bbox = compute_bounding_box(mask)
    if bbox is None:
        print("No foreground found in new x-ray.")
        return None, None
    x0, y0, w0, h0 = bbox
    cropped_xray = new_xray[y0:y0 + h0, x0:x0 + w0]

    target_x, target_y, target_w, target_h = region
    scale = min(target_w / w0, target_h / h0)
    new_w = int(w0 * scale)
    new_h = int(h0 * scale)

    resized_roi = cv2.resize(cropped_xray, (new_w, new_h))

    center_target_x = target_x + target_w // 2
    center_target_y = target_y + target_h // 2
    top_left_x = center_target_x - new_w // 2
    top_left_y = center_target_y - new_h // 2

    return resized_roi, (top_left_x, top_left_y)

def get_region_box(predicted_labels, region_map):
    x = []
    y = []
    for label in predicted_labels:
        if label not in region_map or region_map[label] is None:
            print(f"Label '{label}' not found in region map.")
            continue
        x.append(region_map[label][0])
        x.append(region_map[label][0] + region_map[label][2])
        y.append(region_map[label][1])
        y.append(region_map[label][1] + region_map[label][3])

    target_x = min(x)
    target_y = min(y)
    target_w = max(x) - target_x
    target_h = max(y) - target_y

    return target_x, target_y, target_w, target_h

def overlay(new_xray, predicted_labels, region_map, template_path,
            white_thresh=240, deskew=False, alpha=0.7):

    full_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if full_template is None:
        print("Error: Could not load template image.")
        return None

    overlay_template = full_template.copy()

    for label in predicted_labels:
        if label not in region_map or region_map[label] is None:
            print(f"Label '{label}' not found in region map.")
            continue

        region = region_map[label]
        roi, top_left = overlay_for_region(new_xray, region, deskew=deskew, white_thresh=white_thresh)
        if roi is None:
            print(f"Overlay creation failed for label '{label}'.")
            continue

        roi_h, roi_w = roi.shape
        x0, y0 = top_left

        if (x0 < 0 or y0 < 0 or x0 + roi_w > overlay_template.shape[1] or
                y0 + roi_h > overlay_template.shape[0]):
            print(f"Overlay for label '{label}' is out of bounds, skipping.")
            continue

        mask = create_foreground_mask_white_bg(roi, white_thresh=white_thresh)

        roi_float = roi.astype(np.float32)
        template_crop = overlay_template[y0:y0 + roi_h, x0:x0 + roi_w].astype(np.float32)

        blended = template_crop.copy()

        fg = (mask > 0)
        blended[fg] = alpha * roi_float[fg] + (1 - alpha) * template_crop[fg]

        blended = np.clip(blended, 0, 255).astype(np.uint8)

        overlay_template[y0:y0 + roi_h, x0:x0 + roi_w] = blended

    return overlay_template



def find_best_transform(xray, template, scale_range=(0.8, 2.0), angle_range=(0, 90),
                       scale_steps=13, trans_steps=20, angle_steps=90):
    best_score = -1
    best_params = None
    #Some contrasting, attempt to isolate the bones
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(xray)
    # Step 2: Optional denoising / smoothing
    #blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    # Step 3: Apply adaptive thresholding (or Otsu's thresholding)
    _, bone_mask = cv2.threshold(xray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Invert if needed (bones should be white)
    #bone_mask = cv2.bitwise_not(bone_mask)

    # Step 5: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 6: Mask the original image (if you want to preserve grayscale bone values)
    xray = cv2.bitwise_and(xray, xray, mask=bone_mask)

    # Create search range parameters
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = np.linspace(angle_range[0], angle_range[1], angle_steps)

    # Brute force search method
    for scale in scales:
        for angle in angles:
            # Rotate and scale image
            rotated = rotate_image(xray, angle, scale)
            #print(f"rotated: shape={rotated.shape}, dtype={rotated.dtype}")
            #print(f"template: shape={template.shape}, dtype={template.dtype}")
            reflections = [rotated, cv2.flip(rotated, 0), cv2.flip(rotated, 1), cv2.flip(rotated, -1)]  # No flip, vertical, horizontal, both

            for i, reflected in enumerate(reflections):
                # Slide transformed image across template, comparing normed cross-correlation
                res = cv2.matchTemplate(reflected, template, cv2.TM_CCORR_NORMED)
                display_image(template)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # If max objective function for image is greater than best, set as best
                if max_val > best_score:
                    best_score = max_val
                    tx, ty = max_loc
                    best_params = (tx, ty, angle, scale, i)

    return best_params

# Crops out background of image
def crop_xray(xray, white_thresh=220):
    mask = create_foreground_mask_white_bg(xray, white_thresh)
    bbox = compute_bounding_box(mask)

    x0, y0, w0, h0 = bbox
    cropped_xray = xray[y0:y0 + h0, x0:x0 + w0]

    return cropped_xray

def crop_xray_f32(xray, white_thresh=0.9):
    mask = create_foreground_mask_white_bg_f32(xray, white_thresh)
    bbox = compute_bounding_box(mask)

    x0, y0, w0, h0 = bbox
    cropped_xray = xray[y0:y0 + h0, x0:x0 + w0]

    return cropped_xray

def is_extremity(predicted_labels):
    for label in predicted_labels:
        if (label == "Abdomen" or label == "Chest" or label == "Clavicles" or label == "Thoracic Spine" or 
            label == "Pelvis" or label == "Hip" or label == "Lumbar Spine" or 
            label == "Cervical Spine" or label == "Shoulder"):
            return False
        
    return True

def find_best_transform_sitk(xray, template, is_extremity, angle_range=(0, 90), angle_step=3):
    
    template_sitk = sitk.GetImageFromArray(template.astype(np.float32))

    best_metric = float("inf")
    best_transform = None
    best_aligned = None
    if is_extremity:
        flip_variants = [
            ("none", xray),
            ("flip_x", np.flip(xray, axis=0)),
            ("flip_y", np.flip(xray, axis=1)),
            ("flip_xy", np.flip(np.flip(xray, axis=0), axis=1))
        ]
    else:
        flip_variants = [
            ("none", xray),
            ("flip_y", np.flip(xray, axis=1)),
        ]
    for angle_deg in range(angle_range[0], angle_range[1], angle_step):
        angle_rad = np.deg2rad(angle_deg)
        
        for flip_type, flipped_xray in flip_variants:
        # include reflections
            xray_sitk = sitk.GetImageFromArray(flipped_xray.astype(np.float32))
            xray_sitk = sitk.Shrink(xray_sitk, [2, 2])
            base_transform = sitk.Similarity2DTransform()
            base_transform.SetAngle(angle_rad)

            # Apply reflection by scaling
            matrix = np.array(base_transform.GetMatrix()).reshape((2, 2))
            base_transform.SetMatrix(matrix.ravel())

            init_transform = sitk.CenteredTransformInitializer(
                template_sitk, xray_sitk,
                base_transform,
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

            # Registration setup
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetInitialTransform(init_transform, inPlace=False)
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=200,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            try:
                final_transform = registration_method.Execute(template_sitk, xray_sitk)
                metric = registration_method.GetMetricValue()

                if metric < best_metric:
                    best_metric = metric
                    if isinstance(final_transform, sitk.AffineTransform):
                        best_transform = sitk.AffineTransform(final_transform)
                    else:
                        # Try to extract the affine part from a composite transform
                        for i in range(final_transform.GetNumberOfTransforms()):
                            sub_tfm = final_transform.GetNthTransform(i)
                            if isinstance(sub_tfm, sitk.AffineTransform):
                                best_transform = sitk.AffineTransform(sub_tfm)
                                break

                    # Resample best alignment
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(template_sitk)
                    resampler.SetInterpolator(sitk.sitkLinear)
                    resampler.SetTransform(final_transform)
                    aligned_xray = resampler.Execute(xray_sitk)
                    best_aligned = sitk.GetArrayFromImage(aligned_xray)

            except Exception as e:
                print(f"Registration failed at angle {angle_deg}°: {e}")
                continue

    return best_transform, best_aligned




def overlay_cc_sitk(new_xray, predicted_labels, region_map, template_path,
            white_thresh=240, alpha=0.7):
    full_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if full_template is None:
        print("Error: Could not load template image.")
        return None

    is_ex = is_extremity(predicted_labels)
    #full_template = blur_then_sharpen_image(full_template)
    #full_template = apply_clahe(full_template)
    # new_xray = cv2.GaussianBlur(new_xray, (5, 5), 0)
    overlay_template = full_template.copy()

    # Get x, y, w, h of target region(s)
    target_x, target_y, target_w, target_h = get_region_box(predicted_labels, region_map)
    
    # Crop out background of x-ray
    cropped = crop_xray_f32(new_xray)
    h, w = cropped.shape[:2]
    cropped = 1.0 - cropped
    cropped = crop_xray_f32(cropped)
    
    cropped = blur_then_sharpen_image(cropped)
    cropped = apply_clahe(cropped)
    
    
    # Resize image to be smaller than target region
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)    
    resized_xray = cv2.resize(cropped, (new_w, new_h))
    #display_image(resized_xray)
    # Get target region from template
    target_region = overlay_template[target_y:target_y + target_h, target_x:target_x + target_w]
    
    # Find parameters of best transform
    _, registered = find_best_transform_sitk(resized_xray, target_region, is_ex)
    reg_h, reg_w = registered.shape[:2]
    registered_float = registered.astype(np.float32)
    #display_image(registered)
    # Compute center-aligned position for overlay
    center_target_x = target_x + target_w // 2
    center_target_y = target_y + target_h // 2
    x0 = center_target_x - reg_w // 2
    y0 = center_target_y - reg_h // 2

    # Bound check to avoid indexing errors
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(x0 + reg_w, overlay_template.shape[1])
    y1 = min(y0 + reg_h, overlay_template.shape[0])

    # Crop registered image to fit within template bounds (if necessary)
    dx = x1 - x0
    dy = y1 - y0
    registered_cropped = registered_float[:dy, :dx]
    template_crop = overlay_template[y0:y1, x0:x1].astype(np.float32)

    # Create mask to ignore black background (using intensity threshold)
    # This will treat very dark pixels as background and avoid blending them
    black_thresh = 5  # You can tweak this value
    fg = registered_cropped > black_thresh

    # Perform alpha blending only where foreground is present
    blended = template_crop.copy()
    blended[fg] = alpha * registered_cropped[fg] + (1 - alpha) * template_crop[fg]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Apply blended patch to overlay template
    overlay_template[y0:y1, x0:x1] = blended

    return overlay_template


# Creates overlay using normed cross-correlation as objective function
def overlay_cc_normed(new_xray, predicted_labels, region_map, template_path,
            white_thresh=240, alpha=0.7):
    full_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if full_template is None:
        print("Error: Could not load template image.")
        return None
    
    # new_xray = cv2.GaussianBlur(new_xray, (5, 5), 0)
    overlay_template = full_template.copy()

    # Get x, y, w, h of target region(s)
    target_x, target_y, target_w, target_h = get_region_box(predicted_labels, region_map)
    
    # Crop out background of x-ray
    cropped = crop_xray(new_xray)
    h, w = cropped.shape[:2]
    cropped = cv2.bitwise_not(cropped)
    cropped = blur_then_sharpen_image(cropped)
    cropped = apply_clahe(cropped)

    # Resize image to be smaller than target region
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)    
    resized_xray = cv2.resize(cropped, (new_w, new_h))

    # Get target region from template
    target_region = overlay_template[target_y:target_y + target_h, target_x:target_x + target_w]
    #display_image(target_region)
    #display_image(cropped)
    # Find parameters of best transform
    tx, ty, angle, scale, reflect_index = find_best_transform(resized_xray, target_region)

    # Find center of image for rotation
    h, w = resized_xray.shape[:2]
    center = (w // 2, h // 2)
    
    # Apply affine transformation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(resized_xray, M, (w, h))
    
    #Apply reflection
    reflections = [rotated, cv2.flip(rotated, 0), cv2.flip(rotated, 1), cv2.flip(rotated, -1)]
    reflected = reflections[reflect_index]
    
    registered = reflected 
    reg_h, reg_w = registered.shape[:2]

    # Find center of target region
    center_target_x = target_x + target_w // 2
    center_target_y = target_y + target_h // 2
    x0 = center_target_x - new_w // 2
    y0 = center_target_y - new_h // 2

    # Crop out region of template to be blended
    registered_float = registered.astype(np.float32)
    template_crop = overlay_template[y0 + ty:y0 + reg_h + ty, x0 + tx:x0 + reg_w + tx].astype(np.float32)

    blended = template_crop.copy()
    
    mask = create_foreground_mask_white_bg(registered, white_thresh)
    fg = (mask > 0)

    # Apply alpha blending with registered image and region of template
    blended[fg] = alpha * registered_float[fg] + (1 - alpha) * template_crop[fg]

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    overlay_template[y0 + ty:y0 + reg_h + ty, x0 + tx:x0 + reg_w + tx] = blended

    return overlay_template