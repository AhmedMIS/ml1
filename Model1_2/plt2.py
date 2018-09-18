import matplotlib.pyplot as plt
import numpy as np


def x_plot(model):
    plt.plot(model['variable_x_out_acc'])
    plt.plot(model['val_variable_x_out_acc'])
    plt.title('Variable X accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.show()
    plt.savefig("x_plot.pdf")

def y_plot(model):
    plt.plot(model['variable_y_out_acc'])
    plt.plot(model['val_variable_y_out_acc'])
    plt.title('Variable Y accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training_Set', 'Test_Set'], loc='upper left')
    plt.show()
    plt.savefig("y_plot.pdf")

def pre_plot(model):
    plt.plot(model['prediction_Output_acc'])
    plt.plot(model['val_prediction_Output_acc'])
    plt.title('Prediction Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.show()
    plt.savefig("pre_plot.pdf")
