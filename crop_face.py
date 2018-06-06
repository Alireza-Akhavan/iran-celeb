"""Performs face cropping and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2018 http://iran-celeb.ir, https://github.com/Alireza-Akhavan/iran-celeb
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#autour : Atefe Valipour

import os
import sys
from pathlib import Path
from PIL import Image
from shutil import copyfile
import argparse

black_list = ['.txt', '.asp', '.db']


def rename_with_id(filename, i=0):
    name_part = filename.split('.')
    image_id = name_part[0]
    if not i:
        extension = name_part[-1]
    else:
        extension = 'jpg'
    assert int(image_id) > 0, "Invalid file name: %s"(filename)
    if i > 0:
        return image_id + '_' + str(int(i / 4) + 1) + '.' + extension
    else:
        return image_id + '.' + extension


def main(args):
    src_dir = os.path.expanduser(args.input_dir)
    new_dir = os.path.expanduser(args.output_dir)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    actor_ids = os.listdir(src_dir)
    actors = dict()
    for actor_id in actor_ids:
        dir = os.path.join(src_dir, actor_id)
        result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dir) for f in filenames if
                  os.path.splitext(f)[1] not in black_list]
        actors[actor_id] = result

    for actor_id in actor_ids:
        files = actors[actor_id]

        for file in files:
            txt_file_path = os.path.splitext(file)[0] + '.txt'
            my_file = Path(txt_file_path)
            if my_file.is_file():
                with open(txt_file_path, "r") as text_file:
                    lines = text_file.read().split(' ')
                    im = Image.open(file)
                    w = int(im.size[0])
                    h = int(im.size[1])

                for i in range(1, len(lines), 4):
                    lines[i + 3] = lines[i + 3].split('\\', 1)[0]
                    width_box = float(lines[i + 2]) * w
                    string = lines[i + 3].replace('\n', ',')
                    string = string.split(',')[0]
                    height_box = float(string) * h
                    x_start = int(float(lines[i]) * w - width_box / 2)
                    y_start = int(float(lines[i + 1]) * h - height_box / 2)
                    x_end = int(float(lines[i]) * w + width_box / 2)
                    y_end = int(float(lines[i + 1]) * h + height_box / 2)
                    croped = im.crop((x_start, y_start, x_end, y_end))
                    filename = os.path.basename(os.path.normpath(file))
                    if not os.path.exists(new_dir + '/' + actor_id):
                        os.mkdir(new_dir + '/' + actor_id)
                    save_dir = new_dir + '/' + actor_id + '/' + rename_with_id(filename, i)
                    croped.convert('RGB').save(save_dir, 'JPEG')


            else:
                filename = os.path.basename(os.path.normpath(file))
                save_dir = new_dir + '/' + actor_id + '/' + rename_with_id(filename)
                if not os.path.exists(new_dir + '/' + actor_id):
                    os.mkdir(new_dir + '/' + actor_id)
                copyfile(file, save_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory with cleaned downloaded directory and bounding boxes.')
    parser.add_argument('output_dir', type=str, help='Directory with face thumbnails.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))