import os
import shutil

def create_directories(base_path, classes):
    for cls in classes:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

def copy_images_by_class(source_base_dir, target_base_dir, classes):
    for cls in classes:
        class_source_dir = os.path.join(source_base_dir, cls)
        class_target_dir = os.path.join(target_base_dir, cls)
        
        for image_name in os.listdir(class_source_dir):
            src_path = os.path.join(class_source_dir, image_name)
            dst_path = os.path.join(class_target_dir, image_name)
            shutil.copy(src_path, dst_path)

# Define paths
train_base_dir = '../data/train/images'
val_base_dir = '../data/validation/images'

# Define target directories
train_target_dir = '../data/preprocessed/train'
val_target_dir = '../data/preprocessed/validation'

# Define defect classes
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Create subdirectories for each class in the target directories
create_directories(train_target_dir, classes)
create_directories(val_target_dir, classes)

# Copy images to target directories based on their class
copy_images_by_class(train_base_dir, train_target_dir, classes)
copy_images_by_class(val_base_dir, val_target_dir, classes)

print("Data organization completed.")
