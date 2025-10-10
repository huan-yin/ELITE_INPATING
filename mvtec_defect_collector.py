import os
import shutil

def extract_mvtec_defects(source_root, img_dst_root, mask_dst_root):
    """
    Adapt to new directory structure: Extract all non-good defect images (from test) and masks (from ground_truth) in the MVTec dataset
    
    Parameters:
    source_root: Root path of the MVTec dataset (e.g., /data1/lxy/.../mvtec)
    img_dst_root: Path to store new defect images (e.g., ./defect_images)
    mask_dst_root: Path to store new defect masks (e.g., ./defect_masks)
    """
    # 1. Create destination folders if they don't exist
    os.makedirs(img_dst_root, exist_ok=True)
    os.makedirs(mask_dst_root, exist_ok=True)
    
    # Statistical variables
    total_img = 0  # Total number of defect images
    total_mask = 0 # Total number of defect masks
    category_stats = {}  # Statistics by category: {category: {defect_type: (num_images, num_masks)}}
    
    # 2. Iterate through all object category folders (e.g., bottle, cable)
    for category in os.listdir(source_root):
        category_path = os.path.join(source_root, category)
        if not os.path.isdir(category_path):  # Skip non-folders (e.g., README)
            continue
        
        # Initialize statistics for the current category
        category_stats[category] = {}
        
        # 3. Define paths for test (image source) and ground_truth (mask source) of the current category
        test_path = os.path.join(category_path, "test")
        gt_path = os.path.join(category_path, "ground_truth")
        
        # Check if test and ground_truth exist
        if not os.path.exists(test_path):
            print(f"Warning: {category} has no test folder, skipping this category")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: {category} has no ground_truth folder, skipping this category")
            continue
        
        # 4. Iterate through all defect types under test (exclude 'good')
        for defect_type in os.listdir(test_path):
            if defect_type == "good":  # Exclude non-defective samples
                continue
            
            # Define source paths for images and masks of the current defect type
            img_src_dir = os.path.join(test_path, defect_type)  # test/defect_type (stores images)
            mask_src_dir = os.path.join(gt_path, defect_type)   # ground_truth/defect_type (stores masks)
            
            # Check if the image source directory exists
            if not os.path.isdir(img_src_dir):
                print(f"Warning: {category}/test/{defect_type} is not a folder, skipping")
                continue
            
            # 5. Read image files (only keep .png/.jpg, default formats for MVTec)
            img_files = [f for f in os.listdir(img_src_dir) 
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not img_files:
                print(f"Warning: {category}/test/{defect_type} has no valid image files, skipping")
                continue
            
            # 6. Read mask files (corresponding to image paths)
            mask_files = []
            if os.path.isdir(mask_src_dir):
                mask_files = [f for f in os.listdir(mask_src_dir) 
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if not mask_files:
                    print(f"Warning: {category}/ground_truth/{defect_type} has no valid mask files, only extracting images")
            else:
                print(f"Warning: {category}/ground_truth/{defect_type} does not exist, only extracting images")
            
            # Record statistics for the current defect type
            img_cnt = len(img_files)
            mask_cnt = len(mask_files)
            category_stats[category][defect_type] = (img_cnt, mask_cnt)
            total_img += img_cnt
            total_mask += mask_cnt
            
            # 7. Copy images to destination folder (rename to avoid conflicts)
            for img_file in img_files:
                new_img_name = f"{category}_{defect_type}_{img_file}"  # category_defectType_originalFileName
                img_src = os.path.join(img_src_dir, img_file)
                img_dst = os.path.join(img_dst_root, new_img_name)
                shutil.copy2(img_src, img_dst)  # copy2 preserves file metadata (e.g., creation time)
            
            # 8. Copy masks to destination folder (if any)
            for mask_file in mask_files:
                new_mask_name = f"{category}_{defect_type}_{mask_file}"  # Consistent with image naming convention
                mask_src = os.path.join(mask_src_dir, mask_file)
                mask_dst = os.path.join(mask_dst_root, new_mask_name)
                shutil.copy2(mask_src, mask_dst)
    
    # 9. Output final statistics
    print("\n" + "="*60)
    print("MVTec defect data extraction completed (adapted to test/ground_truth structure)!")
    print(f"Total: {total_img} defect images, {total_mask} defect masks")
    print("\nDetailed statistics by category:")
    for category, defects in category_stats.items():
        if not defects:  # Skip categories with no valid defects
            continue
        print(f"\n【{category}】")
        for defect_type, (img_cnt, mask_cnt) in defects.items():
            print(f"  {defect_type:15} | Images: {img_cnt:3} | Masks: {mask_cnt:3}")
    print("="*60)


# -------------------------- Configure Paths --------------------------
# 1. Source dataset root path (path given in the problem)
SOURCE_ROOT = "datasets/mvtec"
# 2. New path to store defect images (customizable, e.g., placed at the same level as the source dataset)
IMG_DST_ROOT = "datasets/mvtec_defect_collection/images"
# 3. New path to store defect masks (customizable, separate from the image path)
MASK_DST_ROOT = "datasets/mvtec_defect_collection/masks"
# ------------------------------------------------------------------------

# Execute extraction (just run the script directly)
if __name__ == "__main__":
    extract_mvtec_defects(SOURCE_ROOT, IMG_DST_ROOT, MASK_DST_ROOT)