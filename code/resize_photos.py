from PIL import Image
import os, sys

PATH_NAME = "/Users/theayoad/Desktop"

def resize_all_images_in_dir(class_name, files_in_dir_list, final_size):
    for file_name_index, file_name in enumerate(files_in_dir_list):
         print('\n{} file_name: {}'.format(file_name_index, file_name))
         if file_name == '.DS_Store':
             continue
         print('PATH_NAME/class_name/file_name: {}/{}/{}'.format(PATH_NAME, class_name, file_name))
         if os.path.isfile('{}/{}/{}'.format(PATH_NAME, class_name, file_name)):
             im = Image.open('{}/{}/{}'.format(PATH_NAME, class_name, file_name))
             f, e = os.path.splitext('{}/{}/{}'.format(PATH_NAME, class_name, file_name))
             size = im.size
             print('original image size: {}'.format(size))
             ratio = float(final_size) / max(size)
             new_image_size = tuple([int(x*ratio) for x in size])
             im = im.resize(new_image_size, Image.ANTIALIAS)
             new_im = Image.new("RGB", (final_size, final_size))
             new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
             new_im.save('{}_img_{}_resized.jpg'.format(class_name.lower(), file_name_index), 'JPEG', quality=90)


# class_names = ['Wooden-Spoon', 'Wooden-Fork', 'Metal-Spoon', 'Metal-Fork', 'Metal-Spatula']
class_names = ['Wooden-Fork', 'Wooden-Fork2']
for class_name in class_names:
    files_in_dir_list = os.listdir('{}/{}'.format(PATH_NAME, class_name))
    resize_all_images_in_dir(class_name=class_name, files_in_dir_list=files_in_dir_list, final_size=256)
