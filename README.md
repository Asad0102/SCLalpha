# SCLalpha

(I) Make sure to have the folders named ‘labels’ and ‘norm_data’ from the drive (audio_data → transformed_recorded_data1022) and save it into the existing data folder.

(II) To convert the data run:
        prepro.py <your_argument>  (with argument as a string that is included in the data file)
      e.g
        - prepro.py key
        - prepro.py knock
        - prepro.py laugh
        - prepro.py ring
        - prepro.py test (To run all)

(III) To run the inference:
        run.py
