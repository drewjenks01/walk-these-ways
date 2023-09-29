import gzip
import pickle as pkl
import pickletools
import pathlib
import logging

def update_demos(self):
        logging.info('Updating demos...demo',self.run_num)
        log_files = sorted([str(p) for p in pathlib.Path(self.log_path).glob("*.pkl")],key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))
    
        for log in log_files:
            with open(log, 'rb') as f:
                demo = pkl.load(f)

            with gzip.open(log, "wb") as f:
                pickled = pkl.dumps(demo)
                optimized_pickle = pickletools.optimize(pickled)
                f.write(optimized_pickle)