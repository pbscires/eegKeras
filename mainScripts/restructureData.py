import os
import numpy as np

if __name__ == "__main__":
    # for filename in os.listdir("D:\\Documents\\csv_chb_fft_lstmpredicted"):
    filename = "FFT.chb01_02.edf.csv"
    array = np.genfromtxt("D:\\Documents\\csv_chb_fft_lstmpredicted\\"+filename, delimiter=',')
    print(array)
    np.delete(array, 0, 1)
    print(array)
    # np.savetxt("D:\\Documents\\csv_chb_fft_lstmpredicted\\"+filename, array, delimiter=',')