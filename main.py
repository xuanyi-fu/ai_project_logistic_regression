from auto_model import auto_logistic_regression
from utils import *
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

import PySimpleGUI as sg
import numpy as np
import matplotlib
matplotlib.use('Agg')

CUSTOMIZED_WEIGHT_REG_FUNC = None
REG_FUNC = auto_logistic_regression

EVENT_DATA_POINT_FILE_PATH = '-DataPointCSVFilePath-'
EVENT_LABEL_FILE_PATH = '-LabelCSVFilePath-'
EVENT_CUSTOMIZED_WEIGHT_FILE_PATH = '-CustomizedWeightFilePath-'
EVENT_USE_CUSTOMIZED_WEIGHT = '-UseCustomizedWeight-'
EVENT_PLOT_DATA_POINT = '-PlotDataPoint-'
EVENT_CANVAS = '-Canvas-'
EVENT_REG_STRENGTH = '-RegStrength-'
EVENT_RUN = '-Run-'
EVENT_TEST_SIZE_SLIDER = '-TestSize-'

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 1000
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500
FILE_PATH_INPUT_WIDTH = 40
DEFAULT_REG_STRENGTH = 1.0
DEFAULT_TEST_SIZE = 0.2


class Controller:
    dataPointFilePath = ''
    labelFilePath = ''
    useCustomizedWeight = False
    weightFilePath = ''
    regularizationStrength = DEFAULT_REG_STRENGTH
    figure = None
    testSize = DEFAULT_TEST_SIZE

    def CheckNumpyRowsSameLength(matrix: np.ndarray) -> bool:
        num_rows = matrix.shape[0]
        row_length = matrix.shape[1]
        for i in range(num_rows):
            if matrix[i].shape[0] != row_length:
                return False
        return True

    def readCSV2Ndarray(filePath):
        if not os.path.exists(filePath):
            raise Exception("File: " + filePath + " does not exist.")

        npArray = None
        try:
            npArray = np.genfromtxt(filePath, delimiter=",")
            if len(npArray.shape) == 1:
                npArray = np.expand_dims(npArray, axis=1)
        except Exception:
            raise Exception("File: " + filePath +
                            " is not a valid CSV File. (Numpy Parsing Failed)")

        if not Controller.CheckNumpyRowsSameLength(npArray):
            raise Exception("CSV File: " + filePath +
                            " has inconsistent row length.")

        return npArray

    def __init__(self) -> None:
        self.figure = Figure()
        self.figureSubplot = self.figure.add_subplot(111)

    def setTestSize(self, input):
        self.testSize = input

    def setRegularizationStrength(self, input):
        if input == '':
            self.regularizationStrength = DEFAULT_REG_STRENGTH
        else:
            self.regularizationStrength = float(input)

    def readDataPointAndLabel(self):
        self.checkPlotReady()
        dataPointNdarray = Controller.readCSV2Ndarray(self.dataPointFilePath)
        labelNdarray = Controller.readCSV2Ndarray(self.labelFilePath)

        if dataPointNdarray.shape[0] != labelNdarray.shape[0]:
            raise Exception(
                'Number of Data Points does not match Number of Labels.')

        return dataPointNdarray, labelNdarray

    def plotDataPoint(self):
        def pcaReduceDim(ndarray, dim=2):
            pca = PCA(n_components=dim)
            pca.fit(ndarray)
            return pca.transform(ndarray)

        dataPointNdarray, labelNdarray = self.readDataPointAndLabel()

        if (dataPointNdarray.shape[1] > 2):
            dataPointNdarray = pcaReduceDim(dataPointNdarray)

        self.figure.clf()
        scatter = plt.scatter(
            x=dataPointNdarray[:, 0], y=dataPointNdarray[:, 1], c=labelNdarray)
        plt.legend(handles=scatter.legend_elements()[0], labels=['0', '1'])
        return plt.gcf()

    def checkPlotReady(self):
        if self.dataPointFilePath == '':
            raise Exception("Data Point CSV file not assigned.")

        if self.labelFilePath == '':
            raise Exception("Label CSV file not assigned.")

    def checkReady(self):
        self.checkPlotReady()
        if self.useCustomizedWeight and self.weightFilePath == '':
            raise Exception("Weight CSV file not assigned.")

    def generateRegressionInput(self):
        self.checkReady()
        data_point_ndarray = Controller.readCSV2Ndarray(self.dataPointFilePath)
        label_ndarray = Controller.readCSV2Ndarray(self.labelFilePath)
        label_ndarray = np.squeeze(label_ndarray)
        weight_ndarray = None
        if self.useCustomizedWeight:
            weight_ndarray = Controller.readCSV2Ndarray(self.weightFilePath)

        return {OUTPUT_DICT_DATA_POINT_KEY: data_point_ndarray,
                OUTPUT_DICT_LABELS_KEY: label_ndarray,
                OUTPUT_DICT_REG_STRENGTH_KEY: self.regularizationStrength,
                OUTOUT_DICT_WEIGHT_KEY: weight_ndarray,
                OUTPUT_DICT_TEST_SIZE_KEY: self.testSize}


def draw_figure(canvas, figure):
    if not hasattr(draw_figure, 'canvas_packed'):
        draw_figure.canvas_packed = {}
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    if widget not in draw_figure.canvas_packed:
        draw_figure.canvas_packed[widget] = figure
        widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    try:
        draw_figure.canvas_packed.pop(figure_agg.get_tk_widget())
    except Exception as e:
        print(f'Error removing {figure_agg} from list', e)
    plt.close('all')


def main():
    def filePathLayout(text, key):
        return [sg.Text(text), sg.Push(), sg.Input(visible=True, enable_events=True, size=(FILE_PATH_INPUT_WIDTH, 1), key=key), sg.FileBrowse()]
    layout = [
        filePathLayout("Data Points CSV: ", EVENT_DATA_POINT_FILE_PATH),
        filePathLayout("Label CSV: ", EVENT_LABEL_FILE_PATH),
        [sg.HSep()],
        [sg.Checkbox("Use Cutomized Weight",
                     k=EVENT_USE_CUSTOMIZED_WEIGHT, enable_events=True)],
        filePathLayout("Customized Weight CSV: ",
                       EVENT_CUSTOMIZED_WEIGHT_FILE_PATH),
        [sg.HSep()],
        [sg.Text('Regularization Strength'), sg.Input(default_text=str(DEFAULT_REG_STRENGTH), visible=True,
                                                      enable_events=True, size=(int(FILE_PATH_INPUT_WIDTH / 2), 1), key=EVENT_REG_STRENGTH)],
        [sg.Text('Test Size'), sg.Slider(range=(0, 1), default_value=DEFAULT_TEST_SIZE,
                                         resolution=0.01, orientation='h', enable_events=True, key=EVENT_TEST_SIZE_SLIDER)],
        [sg.HSep()],
        [sg.Button('Plot Data Points', enable_events=True, k=EVENT_PLOT_DATA_POINT),
            sg.VSep(),
            sg.Button('Run', enable_events=True, k=EVENT_RUN)],
        [sg.HSep()],
        [sg.Canvas(key=EVENT_CANVAS, size=(CANVAS_WIDTH, CANVAS_HEIGHT),
                   background_color=sg.theme_button_color()[1])]
    ]
    window = sg.Window('Logistic Regression', layout,
                       grab_anywhere=False, finalize=True)
    controller = Controller()
    figure_agg = None
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break
        try:
            if event == EVENT_DATA_POINT_FILE_PATH:
                controller.dataPointFilePath = values[EVENT_DATA_POINT_FILE_PATH]
            elif event == EVENT_LABEL_FILE_PATH:
                controller.labelFilePath = values[EVENT_LABEL_FILE_PATH]
            elif event == EVENT_CUSTOMIZED_WEIGHT_FILE_PATH:
                controller.weightFilePath = values[EVENT_CUSTOMIZED_WEIGHT_FILE_PATH]
            elif event == EVENT_USE_CUSTOMIZED_WEIGHT:
                controller.useCustomizedWeight = values[EVENT_USE_CUSTOMIZED_WEIGHT]
            elif event == EVENT_PLOT_DATA_POINT:
                if figure_agg:
                    delete_figure_agg(figure_agg)
                fig = controller.plotDataPoint()
                figure_agg = draw_figure(window[EVENT_CANVAS].TKCanvas, fig)
            elif event == EVENT_REG_STRENGTH:
                controller.setRegularizationStrength(
                    values[EVENT_REG_STRENGTH])
            elif event == EVENT_TEST_SIZE_SLIDER:
                controller.setTestSize(values[EVENT_TEST_SIZE_SLIDER])
            elif event == EVENT_RUN:
                regInput = controller.generateRegressionInput()
                output = REG_FUNC(regInput)
                result = output[MODEL_OUTPUT_DICT_RESULTS_KEY]
        except Exception as e:
            sg.Popup(str(e), keep_on_top=True)
    window.close()


if __name__ == '__main__':
    main()
