import cv2
import os
import matplotlib.pyplot as plt

def show_sample_faces(base_dir, num_samples=5):
    # Create a figure with subplots for each student
    plt.figure(figsize=(15, 10))
    
    # Get list of student folders (1-6)
    student_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    
    for idx, student_folder in enumerate(student_folders, 1):
        folder_path = os.path.join(base_dir, student_folder)
        images = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        
        # Select sample images evenly spaced
        step = len(images) // num_samples
        sample_images = images[::step][:num_samples]
        
        # Create subplot for current student
        for i, img_name in enumerate(sample_images, 1):
            plt.subplot(len(student_folders), num_samples, (idx-1)*num_samples + i)
            img = cv2.imread(os.path.join(folder_path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            plt.imshow(img)
            plt.axis('off')
            if i == 1:  # Only show student number on first image in row
                plt.title(f'Student {student_folder}')

    plt.tight_layout()
    plt.show()

# Show sample faces from the training data
show_sample_faces('data/train')
