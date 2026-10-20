import concurrent.futures

from .image_loader import *
from .registration import *
from .mirroring import *
from .constants import *

# def create_site_template(
#     site_folder,
#     n_clusters=2,
#     keep_ratio=0.3,
#     second_pass=True,
#     resize_dim=(64, 64)
# ):
#     images = load_xray_images(site_folder)
#     if not images:
#         print(f"No images found in {site_folder}")
#         return None
#     if len(images) == 1:
#         return (images[0] * 255).astype(np.uint8)
#
#     indices = cluster_images(images, resize_dim=resize_dim, n_clusters=n_clusters)
#     selected = [images[i] for i in indices]
#     print(f"{os.path.basename(site_folder)}: selected {len(selected)} from cluster out of {len(images)}")
#
#     fixed = selected[0]
#     pass1_results = []
#     for idx, img in enumerate(selected, start=1):
#         reg, metric = robust_register_image(fixed, img)
#         if reg is not None and metric is not None:
#             pass1_results.append((reg, metric))
#         else:
#             print(f"  Registration failed for image {idx} in pass 1")
#
#     if not pass1_results:
#         print(f"All images failed registration for {site_folder}")
#         return None
#
#     pass1_results.sort(key=lambda x: x[1])  # sort by metric
#     keep_count = max(1, int(len(pass1_results) * keep_ratio))
#     kept_pass1 = pass1_results[:keep_count]
#     print(f"  {site_folder}: pass1 => kept {keep_count}/{len(pass1_results)} images (best metrics)")
#
#     pass1_imgs = [r[0] for r in kept_pass1]
#     preliminary_template = create_xray_template_median(pass1_imgs).astype(np.float32) / 255.0
#
#     if second_pass:
#         pass2_results = []
#         for idx, (reg_img, _) in enumerate(kept_pass1, start=1):
#             reg, metric = robust_register_image(preliminary_template, reg_img)
#             if reg is not None and metric is not None:
#                 pass2_results.append(reg)
#             else:
#                 print(f"  Registration failed for pass2 image {idx}")
#
#         if not pass2_results:
#             print(f"All pass2 images failed re-registration for {site_folder}")
#             return None
#
#         final_template = create_xray_template_median(pass2_results)
#     else:
#         final_template = (preliminary_template * 255).astype(np.uint8)
#
#     # folder_name = os.path.basename(site_folder)
#     # if folder_name in DUPLICATE_SITES:
#     #     h_temp, w_temp = final_template.shape
#     #     final_template = final_template[:, :w_temp // 2]
#     #     print(f"{folder_name}: cropped to left half to remove duplicate content.")
#
#     return final_template

# def generate_site_templates(sorted_dir):
#     site_templates = {}
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_map = {}
#         for site in SITE_LABELS:
#             folder = os.path.join(sorted_dir, site)
#             if not os.path.exists(folder):
#                 print(f"Folder not for: {site} at {folder}")
#                 continue
#             print(f"Submitting template creation for {site}")
#             fut = executor.submit(
#                 create_site_template,
#                 folder,
#                 n_clusters=2,
#                 keep_ratio=0.3,
#                 second_pass=True
#             )
#             future_map[fut] = site
#         for fut in concurrent.futures.as_completed(future_map):
#             s = future_map[fut]
#             try:
#                 tmpl = fut.result()
#                 if tmpl is not None:
#                     print(f"Template created for site: {s}, shape={tmpl.shape}, min={tmpl.min()}, max={tmpl.max()}")
#                     site_templates[s] = tmpl
#             except Exception as e:
#                 print(f"Error processing site {s}: {e}")
#
#     site_templates = mirror_arm_leg_templates(site_templates)
#     return site_templates