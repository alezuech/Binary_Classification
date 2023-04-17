############  CONTENT  ##########################

This folder contains 6 files other than READ_ME.txt:
    - Task_ZUECH.pdf: it contains a description of the task, some figures and the best experimental results
    - object.csv: contains the data used for training and testing.
    - plots.py: python file used to obtain the figures shown in the main pdf document.
    - SVM.py: python file that builds a SVM and tests it with multiple hyper-parameters combinations.
    - MLP.py: python file that builds a MLP and tests it with multiple hyper-parameters combinations.
    - runner.sh: main runner that can be used to install some of the necessary packages
                 Then, runner.sh runs plots.py, SVM.py and MLP.py


############  PACKAGES  ##########################

This project uses:  Python 3.8.10
                    tensorflow-gpu 2.9.1 
                    scikit-learn 1.2.2
                    pandas 1.4.3 
