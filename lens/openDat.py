from numpy import loadtxt

def openDatFile(path='0',skRows=6):
    if path=='0': #if no path is given it opens a window to ask for the file
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        return loadtxt(file_path,skiprows=skRows)
    elif isinstance(path,str): #if one path is given it opens it
        return loadtxt(path,skiprows=skRows)
    elif isinstance(path,list): #if more than one path is given it opens them and returns a set of arrays
        data = list()
        for i in range(len(path)):
            data.append( loadtxt(path[i],skiprows=skRows) )
        return data
    return -1
