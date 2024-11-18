# Copyright (c) 2024 Manuel Rubio

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import cv2
import os
from threading import Thread
from typing import List
import logging


def apply_clahe(paths : List[str], out_dir, extension='.png', thread_name=0) -> bool:
    result = False

    for path in paths:

        if thread_name is not None:
            logging.info(f'{thread_name}:Applying CLAHE on "{path}"')

        basename = os.path.basename(path).replace(extension, '__CLAHE' + extension)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(image)

        tree = path.split('/')[1:-1]
        out_tree = os.path.join(out_dir, *tree)
        
        os.makedirs(out_tree, exist_ok=True)

        outpath = os.path.join(out_tree, basename)
        result = cv2.imwrite(outpath, enhanced_image)

        logging.info(f'{thread_name}:"{basename}" saved on "{outpath}".')

    return result

def _list_files(dir: str):
    for root, dirs, files in os.walk(dir):
        for filename in dirs + files:
            yield os.path.join(root, filename)

def _find(dir: str, ext: str) -> List[str]:
    return [_ for _ in _list_files(dir) if _.endswith(ext)]

def _distribute_work(data: List[str], num_threads: int):
    distributed_work = []
    chunk_size = len(data) // num_threads

    for _ in range(num_threads):
        distributed_work.append(data[:chunk_size])
        data = data[chunk_size:]

    distributed_work[-1].extend(data)
    return distributed_work

def main(args):
    logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.INFO,
                        datefmt='%H:%M:%S')

    indir = args.INPUT_DIRECTORY[0]
    outdir = args.OUTPUT_DIRECTORY[0]
    num_threads = 1

    ext = 'png'
    if args.extension is not None:
        ext = args.extension[0]

    paths = _find(indir, f'.{ext.lower()}')
    ext = f'.{ext.lower()}'    

    # Doing the work using thread-based parallelism
    if args.threads is not None: 
        num_threads = args.threads[0]
        threads = []
        batches = _distribute_work(paths, num_threads)

        for tid in range(num_threads):
            t = Thread(target=apply_clahe, args=[batches[tid], outdir],
                       kwargs={'extension': ext, 'thread_name': tid})
            t.name = tid
            logging.info(f"Starting Thread-{t.name} on chunk no. {tid + 1} of {num_threads}...")
            threads.append(t)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
    else: # Using a single thread of execution
        apply_clahe(paths, outdir, extension=ext)

    logging.info(f'CLAHE sucessfully applied to {len(_find(outdir, ext))} images.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_DIRECTORY', nargs=1, type=str)
    parser.add_argument('OUTPUT_DIRECTORY', nargs=1, type=str)
    parser.add_argument('-e', '--extension', nargs=1, type=str, required=False)
    parser.add_argument('-t', '--threads', nargs=1, type=int, required=False)

    args = parser.parse_args()
    main(args)
