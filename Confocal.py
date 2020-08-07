from __future__ import print_function, absolute_import, division

import numpy
from numpy import ma
import pickle
import Fit
import time
import logging

# enthought library imports
from traits.api import SingletonHasTraits, Instance, Property, Int, Float, Range, \
    Bool, Array, Str, Enum, Button, Tuple, List, on_trait_change, \
    cached_property, DelegatesTo, Trait, Event
from traitsui.api import View, Item, HGroup, VGroup, Tabbed, EnumEditor, ButtonEditor
from enable.api import ComponentEditor
from chaco.api import Plot, ScatterPlot, CMapImagePlot, ArrayPlotData, \
    ColorBar, LinearMapper, DataView, \
    LinePlot, HPlotContainer, VPlotContainer, PlotGraphicsContext
# from chaco.tools.api import ZoomTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
# customized zoom tool to keep aspect ratio
from utility import AspectZoomTool as ZoomTool

import pym8190a

from chaco.default_colormaps import jet
import utility
import threading

from pi3diamond_custom import pi3d


class TakeSlice():
    def __getitem__(self, index):
        return index


take_slice = TakeSlice()


class line_array(numpy.ndarray):
    # 3-dimensional linespace aquivalent with additional paramter front_samples, post_samples and positive_direction
    # if passing front_samples or post_samples: an array will be produced that has additionaly at front or respectivily end #front_samples or respectivily #post_scamples
    # front and post sample will be same value as start_point respectivily end_point
    # passing positive_direction = False: is aquivalent to line_array(end_point,start_point,resolution)
    def __new__(subtype, start_point, end_point, resolution, front_samples=0, post_samples=0, positive_direction=True):
        if positive_direction:
            first_point = start_point
            second_point = end_point
        else:
            first_point = end_point
            second_point = start_point

        steep = (second_point - first_point) * numpy.linspace(0, 1, resolution) + first_point
        all = numpy.empty((3, resolution + front_samples + post_samples))
        for dimension, dimension_data in enumerate(steep):
            dimension_data = numpy.append(numpy.ones((front_samples)) * dimension_data[0], dimension_data)
            dimension_data = numpy.append(dimension_data, numpy.ones((post_samples)) * dimension_data[-1])
            all[dimension] = dimension_data
        return all


class cross_talk_matrix(list):
    # if called through cross_talk_matrix_instance(array([[vx],[vy],[vz]])) it does
    # 
    #       array([
    #               [dx1(vx) + dx2(vy) + dx2(vz)],
    #               [dy1(vx) + dy2(vy) + dy3(vz)],
    #               [dz1(vx),dz2(vy),dz3(vz)],
    #               ])
    #       =array([[dx1],[dx2],[dx3]])
    #    
    #       while self = [ [dx1,dx2,dx2],
    #                      [dy1,dy2,dy3],
    #                      [dz1,dz2,dz3],
    #                        ]

    def __init__(self, cross_talk_array=[]):
        list.__init__(self, cross_talk_array)

    def __call__(self, velocity):
        shift = numpy.empty((numpy.shape(self)[0], 1))
        for i, line in enumerate(self):
            sum = 0
            for j, component in enumerate(line):
                sum += component(velocity[j])
            shift[i] = sum
        return shift


class interp_orig_lin():
    # 1d line fit of m*x + y and y = 0
    def __init__(self, x_data, y_data):
        self.m = numpy.sum(x_data * y_data) / numpy.sum(x_data ** 2)

    def __call__(self, x_grid):
        return self.m * x_grid


class ConfocalLineScan():
    # Use to perform Confocal line scan
    # To creat instance enter ConfocalLineScan(CountTime,SettlingTime, cross_talk_shift)
    # cross_talk_shift has to be None for calculating no cross talk
    # or an instance of cross_talk_matrix class to compensate cross talk (bad results for high velocities)
    #
    # For Performing a Line scan enter ConfocalLineScanInstance(target_points)
    def __init__(self, CountTime, SettlingTime, cross_talk_shift=None, ):
        self.CountTime = CountTime
        self.cross_talk_shift = cross_talk_shift

        self.SettlingTime = SettlingTime
        if pi3d.nidaq.init_scan(SettlingTime, CountTime) != 0:  # returns 0 if successful
            print('error in nidaq')
            return

    def __call__(self, pos_target):
        start_point = numpy.array([[pos_target[0][0]], [pos_target[1][0]], [pos_target[2][0]]])
        self.set_position(start_point)

        if not self.cross_talk_shift == None:
            pos_target = pos_target + self.cross_talk_shift(numpy.diff(pos_target) / (self.CountTime + self.SettlingTime))

        pos_monitor, counts = pi3d.nidaq.scan_read_line(pos_target)
        counts = counts / 1e3

        dictionary = {
            'pos_monitor': pos_monitor,
            'image': counts,
            'pos_target': pos_target
        }

        return dictionary

    def set_position(self, point):
        pi3d.nidaq.scan_read_line(line_array(point, point, 2, 0, 0, True))

    def scan_read_line(Line):
        pass

    def __del__(self):
        pi3d.nidaq.stop_scan()


class SimulateConfocalLineScan():
    # Use to perform a Simulation of Confocal line scan
    # can be used with old version of data saves (pickle confocal._debbug) standing under data_file_name

    # same useage of instance as ConfocalLineScan instance
    def __init__(self, data_file_name):
        self.cross_talk_shift = False
        self.data = pickle.load(open(data_file_name, 'r'))
        self.data_iterator = self.data[:-1].__iter__()

    def __call__(self, pos_target):
        try:
            line = self.data_iterator.next()
            data = {
                'pos_monitor': numpy.array(line[0]),
                'image': numpy.array(line[2]),
                'pos_target': numpy.array(line[1])
            }
            return data
        except StopIteration:
            raise StopIteration

    def get_measurement_parameter(self):
        if not self.data[0][1][0][0] == self.data[1][1][0][0]:
            bidirectional = True

        post_samples = 0

        x1 = self.data[0][1][0][0]

        if bidirectional and ((len(self.data) - 1) % 2) == 1:
            x2 = self.data[-2][1][0][-1]
        else:
            x2 = self.data[-2][1][0][0]

        y1 = self.data[0][1][1][0]
        y2 = self.data[-2][1][1][0]
        x_resolution = len(self.data[0][1][0])
        y_resolution = len(self.data) - 1

        if x_resolution >= y_resolution:
            resolution = x_resolution
        else:
            resolution = y_resolution

        return (x1, x2, y1, y2, resolution, bidirectional, post_samples)


class SimulateConfocalLineScanScan2dClass():
    # Use to perform a Simulation of Confocal line scan
    # can be used with new version of data saves (pickle confocal.scan_data_container,
    # an instance of Scan2d class) standing under data_file_name

    # same useage of instance as ConfocalLineScan instance
    def __init__(self, data_file_name):
        self.cross_talk_shift = False
        import pickle
        self.data_container = pickle.load(open(data_file_name, 'r'))
        self.data_container_generator = self.data_container.get_scan_lines()

    def __call__(self, pos_target):
        try:
            return self.data_container_generator.next()
        except StopIteration:
            raise StopIteration

    def get_measurement_parameter(self):
        x_resolution = self.data_container.x_resolution
        y_resolution = self.data_container.y_resolution

        if x_resolution >= y_resolution:
            resolution = x_resolution
        else:
            resolution = y_resolution

        return (
            self.data_container.x1,
            self.data_container.x2,
            self.data_container.y1,
            self.data_container.y2,
            resolution,
            self.data_container.bidirectional,
            self.data_container.post_samples
        )


class ScanData():
    def __init__(self, line_resolution, bidirectional=True):
        # fill_dimension determins in which dimension incoming data array are appended.
        # fill_dimension\flip_dimension starts from 1 and is equivalent to the very last array dimension
        self.fill_dimension = 0
        # flip_dimension determins while receiving odd numbered scan line and bidirectional scan, in which dimension incoming data is flipped.
        self.flip_dimension = 1

        self.scan_dimensions = 2

        #
        self.number_pos_dimension = 3
        self.line_resolution = line_resolution
        self.bidirectional = bidirectional

        self.__init_data__()
        self.set_current_index_position()

    def append_data_keyword(self, keyword, shape):
        if not hasattr(self, 'data'):
            self.data = {}

        self.data.update({keyword: numpy.empty(shape)})

    def integrate_data_keyword_list(self, list_instance, shape):
        for i, keyword in enumerate(list_instance):
            self.append_data_keyword(keyword, shape)

    def __init_data__(self, full=True):
        self.init_pos_shape = self.get_index_tuple(self.number_pos_dimension, 0, self.line_resolution)
        self.init_image_shape = self.init_pos_shape[1:]

        self.pos_data_keywords = ['pos_monitor', 'pos_target']
        self.image_data_keywords = ['image']
        if full:
            self.integrate_data_keyword_list(self.pos_data_keywords, self.init_pos_shape)
            self.integrate_data_keyword_list(self.image_data_keywords, self.init_image_shape)

    def append(self, data):

        for i, keyword in enumerate(data):
            temp_new_shape = list(self.data[keyword].shape)
            temp_new_shape[self.get_fill_index()] = 1
            data[keyword].shape = tuple(temp_new_shape)

        if (self.data['image'].shape[self.get_fill_index()] % 2 == 0) or not self.bidirectional:
            for i, variable in enumerate(data):
                self.data[variable] = numpy.append(self.data[variable], data[variable], self.get_fill_index())
        else:
            for i, variable in enumerate(data):
                self.data[variable] = numpy.append(self.data[variable], self.flip_direction(data[variable], self.get_flip_index()), self.get_fill_index())
        self.set_current_index_position()

    def get_scan_lines(self):
        for i in range(self.data['image'].shape[self.get_fill_index()]):
            data_dict = {}
            if self.bidirectional and i % 2 == 0:
                line_slice = take_slice[:]
            else:
                line_slice = take_slice[::-1]

            for j, data_key in enumerate(self.pos_data_keywords):
                data_index = self.get_index_tuple(take_slice[:], i, line_slice)
                data_dict.update({data_key: self.data[data_key][data_index]})

                # if not self.get_fill_index() == -1:
                #    data_dict_shape = data_dict[data_key].shape[:self.get_fill_index()] + data_dict[data_key].shape[self.get_fill_index()+1:]                
                # else:
                #    data_dict_shape = data_dict[data_key].shape[:self.get_fill_index()]

                # print data_dict[data_key].shape, data_dict_shape
                # data_dict[data_key].shape = data_dict_shape

            for j, data_key in enumerate(self.image_data_keywords):
                data_index = self.get_index_tuple(take_slice[:], i, line_slice)[1:]
                data_dict.update({data_key: self.data[data_key][data_index]})

                # if not self.get_fill_index() == -1:
                #    data_dict_shape = data_dict[data_key].shape[:self.get_fill_index()] + data_dict[data_key].shape[self.get_fill_index()+1:]                
                # else:
                #    data_dict_shape = data_dict[data_key].shape[:self.get_fill_index()]

                # data_dict[data_key].shape = data_dict_shape

            yield data_dict

    def reset_index_position(self):
        self.index_position = 0

    def set_current_index_position(self):
        self.index_position = self.data['pos_target'].shape[self.get_fill_index()] - 1

    def start_index_position(self, decrement):
        start_index_position = self.index_position - decrement
        if start_index_position < 0:
            start_index_position = 0
        return start_index_position

    def __getitem__(self, index):
        if hasattr(index, '__iter__'):
            return self.data[index[0]][index[1:]]
        else:
            return self.data[index]

    def getitem(self, index):
        if hasattr(index, '__iter__'):
            fill_dimension_value = index[-1]
            flip_dimension_value = index[-2]
            array_index = list(index[1:])
            array_index[self.fill_dimension - self.scan_dimensions] = fill_dimension_value
            array_index[self.flip_dimension - self.scan_dimensions] = flip_dimension_value
            return self.data[index[0]][array_index]
        else:
            return self.data[index]

    def get_index_tuple(self, dimension, fill, flip):
        index_tuple = list(numpy.empty((3)))
        index_tuple[self.get_fill_index()] = fill
        index_tuple[self.get_flip_index()] = flip
        index_tuple[0] = dimension
        return tuple(index_tuple)

    def get_fill_index(self):
        if not hasattr(self, 'fill_index'):
            self.fill_index = self.fill_dimension - self.scan_dimensions
        return self.fill_index

    def get_flip_index(self):
        if not hasattr(self, 'flip_index'):
            self.flip_index = self.flip_dimension - self.scan_dimensions
        return self.flip_index

    def get_data_item(self, data_key, *args):
        pass

    def flip_direction(self, array, dimension):
        return numpy.swapaxes(numpy.flipud(numpy.swapaxes(array, 0, dimension)), dimension, 0)

    def set_bidirectional(self):
        self.bidirectional = True

    def unset_bidirectional(self):
        self.bidirectional = False

    def is_bidirectional(self):
        return self.bidirectional


class MeasureCrossTalk(ScanData):
    def __init__(self, start_velocity, stop_velocity, resolution):

        self.pos_resolution = 500

        self.iterations = {
            "cross_talk_scan_dimensions": [0, 1, 2],
            "positive_directions": [False, True],
            "velocities": numpy.linspace(start_velocity, stop_velocity, resolution)
        }

        self.iterator_pos = {
            "cross_talk_scan_dimensions": 0,
            "positive_directions": False,
            "velocities": start_velocity
        }

        self.start_velocity = start_velocity
        self.stop_velocity = stop_velocity

        self.low_position = numpy.array([[pi3d.nidaq._scanner_xrange[0]], [pi3d.nidaq._scanner_yrange[0]], [pi3d.nidaq._scanner_zrange[0]]])
        self.high_position = numpy.array([[pi3d.nidaq._scanner_xrange[1]], [pi3d.nidaq._scanner_yrange[1]], [pi3d.nidaq._scanner_zrange[1]]])

        ScanData.__init__(self, self.pos_resolution, bidirectional=False)

    def get_target_data(self):
        start_vector = numpy.array([[0], [1. / 2], [1. / 2]])
        end_vector = numpy.array([[1], [1. / 2], [1. / 2]])

        for self.iterator_pos['cross_talk_scan_dimensions'] in self.iterations['cross_talk_scan_dimensions']:
            for self.iterator_pos['positive_directions'] in self.iterations['positive_directions']:
                for self.iterator_pos['velocities'] in self.iterations['velocities']:
                    start_point = self.high_position * numpy.roll(start_vector, self.iterator_pos['cross_talk_scan_dimensions'])
                    stop_point = self.high_position * numpy.roll(end_vector, self.iterator_pos['cross_talk_scan_dimensions'])
                    yield self.iterator_pos['cross_talk_scan_dimensions'], self.iterator_pos['positive_directions'], self.iterator_pos['velocities'], line_array(start_point, stop_point, self.pos_resolution, 0, 0, self.iterator_pos['positive_directions'])

    def get_scan_parameter(self):
        SettlingTime = 0.00001
        CountTime = numpy.abs((self.high_position[self.dimension][0] - self.low_position[self.dimension][0]) / (self.velocity * ((self.high_position[self.dimension][0] - self.low_position[self.dimension][0]) / (self.high_position[0][0] - self.low_position[0][0])) * self.pos_resolution)) - SettlingTime
        return SettlingTime, CountTime

    def get_cross_talk_shift(self):
        cross_talk_shift = []

        def zero(v):
            return 0

        for dimension in self.iterations['cross_talk_scan_dimensions']:
            velocities = numpy.array([])
            dimension_plus_1 = (dimension + 1) % (len(self.low_position))
            dimension_plus_2 = (dimension + 2) % (len(self.low_position))
            crossing1 = numpy.array([])
            crossing2 = numpy.array([])
            for positive_direction in self.iterations['positive_directions']:
                for velocity in self.iterations['velocities']:

                    if not positive_direction:
                        velocities = numpy.append(velocities, -velocity)
                    else:
                        velocities = numpy.append(velocities, velocity)

                    fill_index = self.get_fill_index(dimension, positive_direction, velocity)
                    reference_point1 = self.data['pos_target'][self.get_index_tuple(dimension_plus_1, fill_index, -1)]
                    reference_point2 = self.data['pos_target'][self.get_index_tuple(dimension_plus_2, fill_index, -1)]

                    crossing1 = numpy.append(crossing1, reference_point1 - self.data['pos_monitor'][self.get_index_tuple(dimension_plus_1, fill_index, -1)])
                    crossing2 = numpy.append(crossing2, reference_point2 - self.data['pos_monitor'][self.get_index_tuple(dimension_plus_2, fill_index, -1)])

                if not positive_direction:
                    velocities = velocities[::-1]
                    crossing1 = crossing1[::-1]
                    crossing2 = crossing2[::-1]
                    velocities = numpy.append(velocities, 0)
                    crossing1 = numpy.append(crossing1, 0)
                    crossing2 = numpy.append(crossing2, 0)

            cross_talk_shift_line = [zero, interp_orig_lin(velocities, crossing1), interp_orig_lin(velocities, crossing2)]
            cross_talk_shift.append(cross_talk_shift_line[-dimension:] + cross_talk_shift_line[:-dimension])

        self.cross_talk_shift = cross_talk_matrix(cross_talk_shift)
        return self.cross_talk_shift

    def get_fill_index(self, move_dimension, positive_direction, velocity):
        len_positive_direction = len(self.iterations['positive_directions'])
        len_velocities = len(self.iterations['velocities'])
        velocity_index = list(self.iterations['velocities']).index(velocity, )
        return (len_positive_direction * len_velocities * move_dimension) + (len_velocities * positive_direction) + velocity_index


class Scan2dData(ScanData):
    def __init__(self, x1, x2, y1, y2, x_resolution, y_resolution, post_samples=59, bidirectional=True, interpolate=True):
        if not interpolate:
            post_samples = 0
        ScanData.__init__(self, x_resolution + post_samples, bidirectional)

        self.interpolation_method = '1d'
        self.interpolation_method_dict = {
            '1d': self.calc_line_interpolation,
            '2d': self.calc_image_griddata,
        }

        interpol_shape = self.get_index_tuple(0, y_resolution, x_resolution)[1:]
        self.data.update({'interpol_image': ma.masked_array(numpy.empty(interpol_shape), True)})

        self.y_resolution = y_resolution
        self.x_resolution = x_resolution

        self.post_samples = post_samples

        # self.Y_grid = numpy.linspace(y1,y2,y_resolution)
        # self.X_grid = numpy.linspace(x1,x2,x_resolution)

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.interpolate = interpolate

    def set_interpolation_method(self, value):
        if value in self.interpolation_method_dict.keys():
            self.interpolation_method = value
        else:
            print('Error: interpolation method has to be ', self.interpolation_method_dict.keys())

    def get_interpolation_method(self):
        return self.interpolation_method_dict[self.interpolation_method]

    def calc_line_interpolation(self):
        from scipy import interp
        start_index_position = self.start_index_position(0)
        act_index = self.data['pos_monitor'].shape[self.get_fill_index()] - 1
        for index in range(start_index_position, act_index + 1):
            counts = self.data['image'][self.get_index_tuple(0, act_index, take_slice[:])[1:]]
            x_pos_monitor = self.data['pos_monitor'][self.get_index_tuple(0, act_index, take_slice[:])]
            interpol_image = ma.masked_array(interp(self.X_grid(), x_pos_monitor, counts), False)
            self.data['interpol_image'][self.get_index_tuple(0, act_index, take_slice[:])[1:]] = interpol_image

        self.set_current_index_position()

    def calc_image_griddata(self):
        from scipy.interpolate import griddata
        from numpy import ma

        start_index_position = self.start_index_position(2)

        TAKEALLSLICE = take_slice[:]
        TAKESHORTSLICE = take_slice[start_index_position:]

        pos_index = self.get_index_tuple(take_slice[:2], TAKESHORTSLICE, TAKEALLSLICE)
        # y_pos_index = self.get_index_tuple(1, TAKESHORTSLICE, TAKEALLSLICE)
        image_index = pos_index[1:]

        if self.data['pos_monitor'].shape[self.get_fill_index()] > 1:

            if self.interpolate:
                x_grid, y_grid = numpy.mgrid[self.x1:self.x2:1j * self.x_resolution, self.y1:self.y2:1j * self.y_resolution]
                pos_monitor = self.data['pos_monitor'][pos_index]
                pos_monitor = pos_monitor.reshape((2, -1))
                pos_monitor = numpy.swapaxes(pos_monitor, 0, 1)
                input_image = self.data['image'][image_index]
                input_image = input_image.reshape(-1)

                image = griddata(
                    pos_monitor,
                    input_image,
                    (x_grid, y_grid),
                    method='linear'
                )
                if not image.__class__.__name__ == 'MaskedArray':
                    image = ma.masked_array(image, False)

            # Generate mask if not interpolating
            else:
                # initialize mask and image
                save_shape = self.data['image'][image_index].shape
                raw_image = self.data['image'][image_index]
                mask = numpy.ones(self.data['interpol_image'].shape, dtype=numpy.bool)
                false_mask = numpy.zeros(raw_image.shape, dtype=numpy.bool)
                image = numpy.zeros_like(self.data['interpol_image'])

                # Generate dynamic index
                mask_image_index = self.get_index_tuple(0, take_slice[start_index_position:start_index_position + save_shape[self.fill_dimension]], TAKEALLSLICE)[1:]

                image[mask_image_index] = raw_image
                mask[mask_image_index] = false_mask

                # generate masked array out of mask matrix and image matrix
                image = ma.masked_array(image, mask)

            self.data['interpol_image'] = self.combine(image, self.data['interpol_image'])
            self.set_current_index_position()

    def linspace(self, start, stop, resolution):
        start = float(start)
        stop = float(stop)
        resolution = float(resolution)
        step = (stop - start) / (resolution - 1)
        if (stop - start) > 0:
            stop_arange = stop + step / 2
        else:
            stop_arange = stop - step / 2
        return numpy.arange(start, stop_arange, step)

    def X_grid(self):
        return self.linspace(self.x1, self.x2, self.x_resolution)

    def Y_grid(self):
        return self.linspace(self.y1, self.y2, self.y_resolution)

    def recalc_image_griddata(self, ):
        self.reset_index_position()
        self.get_interpolation_method()()

    def combine(self, ma1, ma2):
        from numpy import ma
        from copy import deepcopy
        ma2 = deepcopy(ma2)

        ma2.mask = numpy.logical_or(numpy.logical_not(ma1.mask), ma2.mask)

        result = ma.masked_array(ma1.filled(0) + ma2.filled(0), numpy.logical_and(ma1.mask, ma2.mask))

        return result

    def get_interpolation(self):
        return self['interpol_image'].filled(0)

    def get_full_interpolation(self):
        self.recalc_image_griddata()
        return self.get_interpolation()

    def target_cross_lost(self, x, y):
        return x < self.x1 or x > self.x2 or y < self.y1 or y > self.y2

    def get_new_target_cross(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def append(self, data):
        ScanData.append(self, data)
        self.get_interpolation_method()()


class LoadScan2d(Scan2dData):
    def __init__(self, file_name, bidirectional=True, interpolation=True):

        import pickle
        temp = pickle.load(open(file_name, 'r'))

        x1 = temp[0][1][0][0]
        if bidirectional and ((len(temp) - 1) % 2) == 1:
            x2 = temp[-2][1][0][-1]
        else:
            x2 = temp[-2][1][0][0]

        y1 = temp[0][1][1][0]
        y2 = temp[-2][1][1][0]

        x_resolution = len(temp[0][1][0])
        y_resolution = len(temp) - 1

        Scan2dData.__init__(self, x1, x2, y1, y2, x_resolution, y_resolution, bidirectional, interpolation)

        for i, line in enumerate(temp[:-1]):
            data = {
                'pos_monitor': line[0],
                'image': line[2],
                'pos_target': numpy.array(line[1])
            }

            self.append(data)


class Scan2d(Scan2dData):
    # Data Container that manages the data while an 2dimensional Confocal Scan.

    # Instance provides depending on initial scan parameters, for each line scan,
    # the target position data (get_target_data generator).
    # plan_z : initial z position
    # post_samples : how many additional samples are added to the target position data
    # bidirectional : if True, target line scans are alternating into negative and positive directions

    # Instance computes through append method readout delay of the position monitor signal (TimeDelay,
    # CountTime, SettlingTime), saves measured data to itself and interpolats (interpolate = True)
    # the received non uniformly distributed data automaticly to regualr grid (for the plotter)
    def __init__(self, x1, x2, y1, y2, resolution, plane_z, post_samples=59, calc_dz=None, TimeDelay=0, CountTime=None, SettlingTime=None, bidirectional=True, interpolate=True):
        if (x2 - x1) >= (y2 - y1):
            x_resolution = resolution
            y_resolution = int(resolution * (y2 - y1) / (x2 - x1))
        else:
            y_resolution = resolution
            x_resolution = int(resolution * (x2 - x1) / (y2 - y1))

        Scan2dData.__init__(self, x1, x2, y1, y2, x_resolution, y_resolution, post_samples, bidirectional, interpolate)

        self.plane_z = plane_z
        self.TiltCorrection = False
        if not calc_dz == None:
            self.TiltCorrection = True
            self._calc_dz = calc_dz

        self.SettlingTime = SettlingTime
        self.CountTime = CountTime
        self.TimeDelay = TimeDelay

    def append(self, data):
        Scan2dData.append(self, self.get_timedelay_corrected(data))

    def get_target_data(self):
        for i, y in enumerate(self.Y_grid()):
            yield i, line_array(
                start_point=self.project2d(self.x1, y),
                end_point=self.project2d(self.x2, y),
                resolution=self.x_resolution,
                post_samples=self.post_samples,
                positive_direction=not self.bidirectional or (i % 2 == 0 and self.bidirectional),
            )

    def project2d(self, x, y):
        if self.TiltCorrection:
            return numpy.array([[x], [y], [self.plane_z + self._calc_dz(x, y)]])
        else:
            return numpy.array([[x], [y], [self.plane_z]])

    def get_timedelay_corrected(self, data):
        if not self.TimeDelay == 0:
            xyz_monitor = data['pos_monitor']

            from scipy import interp
            time = numpy.arange(len(xyz_monitor[0])) * (self.SettlingTime + self.CountTime)
            xyz_monitor[0] = interp(time, time - self.TimeDelay, xyz_monitor[0])
            xyz_monitor[1] = interp(time, time - self.TimeDelay, xyz_monitor[1])
            xyz_monitor[2] = interp(time, time - self.TimeDelay, xyz_monitor[2])

            data['pos_monitor'] = numpy.array(xyz_monitor)
        return data


class Confocal(SingletonHasTraits, utility.GetSetItemsMixin):
    x = Range(low=pi3d.nidaq._scanner_xrange[0], high=pi3d.nidaq._scanner_xrange[1], value=0.5 * (pi3d.nidaq._scanner_xrange[0] + pi3d.nidaq._scanner_xrange[1]), desc='x [micron]', label='x [micron]', mode='slider')
    y = Range(low=pi3d.nidaq._scanner_yrange[0], high=pi3d.nidaq._scanner_yrange[1], value=0.5 * (pi3d.nidaq._scanner_yrange[0] + pi3d.nidaq._scanner_yrange[1]), desc='y [micron]', label='y [micron]', mode='slider')
    z = Range(low=pi3d.nidaq._scanner_zrange[0], high=pi3d.nidaq._scanner_zrange[1], value=0.5 * (pi3d.nidaq._scanner_zrange[0] + pi3d.nidaq._scanner_zrange[1]), desc='z [micron]', label='z [micron]', mode='slider')

    x1 = Range(low=pi3d.nidaq._scanner_xrange[0], high=pi3d.nidaq._scanner_xrange[1], value=pi3d.nidaq._scanner_xrange[0], desc='x1 [micron]', label='x1', mode='text')
    y1 = Range(low=pi3d.nidaq._scanner_yrange[0], high=pi3d.nidaq._scanner_yrange[1], value=pi3d.nidaq._scanner_yrange[0], desc='y1 [micron]', label='y1', mode='text')
    x2 = Range(low=pi3d.nidaq._scanner_xrange[0], high=pi3d.nidaq._scanner_xrange[1], value=pi3d.nidaq._scanner_xrange[1], desc='x2 [micron]', label='x2', mode='text')
    y2 = Range(low=pi3d.nidaq._scanner_yrange[0], high=pi3d.nidaq._scanner_yrange[1], value=pi3d.nidaq._scanner_yrange[1], desc='y2 [micron]', label='y2', mode='text')
    max_range_button = Button()

    def _max_range_button_fired(self):
        self.x1 = 0
        self.x2 = 100
        self.y1 = 0
        self.y2 = 100

    threads = Trait({})  # it is very important to keep this a trait, otherwise state machine gets messed up,
    count_thread = Trait()

    # since values can be delayed if called from thread

    def __init__(self):
        SingletonHasTraits.__init__(self)
        self.state2method = {'scan': self.scan,
                             'xyz_monitor_scan': self.xyz_monitor_scan,
                             'refocus_crosshair': self.refocus,
                             'refocus_trackpoint': self.track,
                             'periodic_refocus': self.DaemonLoop}
        self.ScanPlot.index_range.on_trait_change(self.set_x1, '_low_value')
        self.ScanPlot.index_range.on_trait_change(self.set_x2, '_high_value')
        self.ScanPlot.value_range.on_trait_change(self.set_y1, '_low_value')
        self.ScanPlot.value_range.on_trait_change(self.set_y2, '_high_value')
        self.on_trait_change(handler=self.set_mesh_and_aspect_ratio, name='X')
        self.on_trait_change(handler=self.set_mesh_and_aspect_ratio, name='Y')
        self.on_trait_change(handler=self.set_cursor_from_position, name='x')
        self.on_trait_change(handler=self.set_cursor_from_position, name='y')
        self.cursor.on_trait_change(handler=self.set_position_from_cursor, name='current_position')
        self.trace_update_count = 0
        self.reset_settings()

    def set_x1(self):
        self.x1 = self.ScanPlot.index_range.low

    def set_x2(self):
        self.x2 = self.ScanPlot.index_range.high

    def set_y1(self):
        self.y1 = self.ScanPlot.value_range.low

    def set_y2(self):
        self.y2 = self.ScanPlot.value_range.high

    def set_mesh_and_aspect_ratio(self):
        self.ScanImage.index.set_data(self.X, self.Y)
        x1 = self.x1 = self.X[0]
        x2 = self.x2 = self.X[-1]
        y1 = self.y1 = self.Y[0]
        y2 = self.y2 = self.Y[-1]
        self.ScanPlot.aspect_ratio = (x2 - x1) / float((y2 - y1))
        self.ScanPlot.index_range.low = x1
        self.ScanPlot.index_range.high = x2
        self.ScanPlot.value_range.low = y1
        self.ScanPlot.value_range.high = y2

    def set_cursor_from_position(self):
        self.cursor.on_trait_change(handler=self.set_position_from_cursor, name='current_position', remove=True)
        self.cursor.current_position = (self.x, self.y)
        self.cursor.on_trait_change(handler=self.set_position_from_cursor, name='current_position')

    def set_position_from_cursor(self):
        self.on_trait_change(handler=self.set_cursor_from_position, name='x', remove=True)
        self.on_trait_change(handler=self.set_cursor_from_position, name='y', remove=True)
        self.x, self.y = self.cursor.current_position
        self.on_trait_change(handler=self.set_cursor_from_position, name='x')
        self.on_trait_change(handler=self.set_cursor_from_position, name='y')

    def load_confocal_image(self, filename):  # load variabel with original flouscence data
        fileobject = open(filename)
        data = pickle.load(fileobject)
        self.X = data['X']
        self.Y = data['Y']
        self.image = data['image']
        self.ScanImage.index.set_data(data['X'], data['Y'])
        self._image_changed()

    def save_confocal_image(self, filename):  # save as variabel with original flouscence data
        fileobject = open(filename, 'w')
        data = {'X': self.X, 'Y': self.Y, 'image': self.image}
        pickle.dump(data, fileobject)

    HighLimit = Float(10000., desc='High Limit of image plot', label='color high', auto_set=False, enter_set=True)
    LowLimit = Float(0., desc='Low Limit of image plot', label='color low', auto_set=False, enter_set=True)
    Autoscale = Button(desc='autoscale', label='Autoscale')

    # tilt correction
    TiltCorrection = Bool()
    r1 = Array(dtype=numpy.float, value=(0., 0., 0.))
    r2 = Array(dtype=numpy.float, value=(0., 0., 0.))
    r3 = Array(dtype=numpy.float, value=(0., 0., 0.))
    x0 = Float(0.5 * (pi3d.nidaq._scanner_xrange[0] + pi3d.nidaq._scanner_xrange[1]))
    y0 = Float(0.5 * (pi3d.nidaq._scanner_yrange[0] + pi3d.nidaq._scanner_yrange[1]))
    ax = Float(0.)
    ay = Float(0.)
    SetR1Button = Button(desc='set R1 to current position', label='Set R1')
    SetR2Button = Button(desc='set R2 to current position', label='Set R2')
    SetR3Button = Button(desc='set R3 to current position', label='Set R3')
    TiltAngleButton = Button(desc='set tilt angle based on R1, R2, R3', label='Set Tilt Angle')
    TiltReferenceButton = Button(desc='set point of zero correction to current position', label='Set Reference')

    resolution = Range(low=1, high=1000, value=100, desc='Number of point in long direction', label='resolution')

    SettlingTime = Range(low=1e-4, high=10, value=0.0001, desc='Settling Time of Scanner [s]', label='Settling Time [s]', mode='text')
    CountTime = Range(low=1e-4, high=10, value=0.002, desc='Count Time of Scanner [s]', label='Count Time [s]', mode='text')

    scan_offset = Int(0, label='Offset during fast scan [steps]', desc='Offset for fast bidirectional scanning [resolution steps]')

    CountTimeMonitor = Range(low=1e-4, high=10, value=0.002, desc='Count Time of Scanner [s]', label='Count Time [s]', mode='text')
    ReadoutDelayMonitor = Range(low=-10.0, high=10.0, value=0.002, desc='Delay time of scanner position readout [s]', label='Readout delay time', mode='text')
    CrossTalkCorrectionMonitor = Bool(False, label='Enable Cross Talk Correction')
    MeasureCrossTalkMonitor = Event
    MeasureCrossTalkLabelMonitor = Str('(Re)Measure cross talk')
    MeasureCrossTalkVelocityMonitor = Range(low=1, high=20000, value=10000, desc='maximum velocity measuring cross talk [micron/s]', label='max velocity')
    MeasureCrossTalkResolutionMonitor = Range(low=1, high=1000, value=10, desc='how many steps measurement in each direction is done', label='steps')
    # TraceSettlingTime = Range(low=1e-3, high=10, value=0.01, desc='Settling Time of Trace [s]', label='Settling Time [s]', mode='text')
    # TraceCountTime = Range(low=1e-3, high=10, value=0.01, desc='Count Time of Trace [s]', label='Count Time [s]', mode='text')

    TraceLength = Range(low=10, high=10000, value=300, desc='Length of Count Trace', label='Trace Length', mode='text')
    CountInterval = Range(low=0.0001, high=10, value=0.02, desc='Count Interval [s]', label='Count Interval [s]', mode='text')
    ReadoutInterval = Range(low=0.01, high=10, value=0.02, label='Readout Interval [s]', desc='Readout interval of data from nidaq memory', mode='text')
    # UpdateInterval = Int(1, label='Update interval', desc='plot update interval / each n-th readout')

    aom_voltage = Range(low=-10., high=-6., value=0., label='Laser Power', mode='slider')

    SaveImage = Button(label='Save Image')
    FilePath = Str('D:\\data\\ScanImages\\')
    FileName = Str('enter filename')

    # Tracker parameters
    XYSize = Range(low=0.1, high=10., value=1., desc='Size of XY Scan [micron]', label='XY Size [micron]', mode='slider')
    XYStep = Range(low=0.001, high=10., value=0.1, desc='Step of XY Scan [micron]', label='XY Step [micron]', mode='slider')
    XYSettlingTime = Range(low=1e-3, high=10, value=0.01, desc='Settling Time of XY Scan [s]', label='XY Settling Time [s]', mode='text')
    XYCountTime = Range(low=1e-3, high=10, value=0.02, desc='Count Time of XY Scan [s]', label='XY Count Time [s]', mode='text')

    ZSize = Range(low=0.1, high=25., value=1., desc='Size of Z Scan [micron]', label='Z Size [micron]', mode='slider')
    ZStep = Range(low=0.01, high=10., value=0.1, desc='Step of Z Scan [micron]', label='Z Step [micron]', mode='slider')
    ZSettlingTime = Range(low=1e-3, high=10, value=0.01, desc='Settling Time of Z Scan [s]', label='Z Settling Time [s]', mode='text')
    ZCountTime = Range(low=1e-3, high=10, value=0.2, desc='Count Time of Z Scan [s]', label='Z Count Time [s]', mode='text')

    # XY offset for refocus, i.e. vertical scanner
    XOffset = Range(low=-1., high=1., value=0, desc='X offset [micron] for refocussing result and for refocussing start position', label='X offset [micron] for refocus', mode='text')
    YOffset = Range(low=-1., high=1., value=0, desc='Y offset [micron] for refocussing result and for refocussing start position', label='Y offset [micron] for refocus', mode='text')

    XYFitMethod = Enum('Maximum', 'Gaussian', 'ignore', desc='Fit Method for XY Scan', label='XY Fit Method')
    ZFitMethod = Enum('Maximum', 'Polynomial(4th-order)', desc='Fit Method for Z Scan', label='Z Fit Method')
    zfit = Float()

    TrackInterval = Range(low=1, high=6000, value=60, desc='Track Interval [s]', label='Track Interval [s]')

    # scan data
    X = Array()
    Y = Array()
    X_monitor = Array()
    Y_monitor = Array()
    image = Array()
    interpolate = Bool(False)

    # trace data
    C = Array()
    T = Array()

    # Tracker data
    XT = Array(value=numpy.arange(0., 1., 0.1), transient=True)
    YT = Array(value=numpy.arange(0., 1., 0.1), transient=True)
    xydata = Array(transient=True)

    ZT = Array(value=numpy.arange(0., 1., 0.1), transient=True)
    zdata = Array(transient=True)

    TrackPoint = Array()
    drift = Array(value=numpy.array(((0, 0, 0),)), transient=True)
    CurrentTrackPoint = Property(trait=Array(), depends_on='TrackPoint')
    CurrentDrift = Property(trait=Array(), depends_on='drift')

    @cached_property
    def _get_CurrentTrackPoint(self):
        if self.TrackPoint is None or len(self.TrackPoint) == 0:
            return self.TrackPoint
        else:
            return self.TrackPoint[-1]

    @cached_property
    def _get_CurrentDrift(self):
        if self.drift is None or len(self.drift) == 0:
            return self.drift
        else:
            return self.drift[-1]

    TargetMode = Bool(False)
    TargetList = List()
    CurrentTargetIndex = Int(value=-1, label='target', desc='Index of current target point')
    CurrentTarget = Property(trait=Tuple(Float, Float), depends_on='CurrentTargetIndex')
    ShowTargets = DelegatesTo('TargetPlot', prefix='visible')
    next_target_btn = Button(label='Next')
    previous_target_btn = Button(label='Prev.')
    remove_duplicate_targets = Button(label='Remove Duplicates')

    # plots
    ScanData = Instance(ArrayPlotData, transient=True)
    ScanPlotContainer = Instance(HPlotContainer, transient=True)
    ScanPlot = Instance(Plot, transient=True)
    ScanImage = Instance(CMapImagePlot, transient=True)
    cursor = Instance(BaseCursorTool, transient=True)
    zoom_tool = Instance(ZoomTool, transient=True)
    TargetPlot = Instance(ScatterPlot, transient=True)

    TracePlot = Instance(Plot, transient=True)
    TraceData = Instance(ArrayPlotData, transient=True)

    TrackImageData = Instance(ArrayPlotData, transient=True)
    TrackLineData = Instance(ArrayPlotData, transient=True)
    TrackImage = Instance(CMapImagePlot, transient=True)
    TrackLine = Instance(LinePlot, transient=True)
    TrackImagePlot = Instance(Plot, transient=True)
    TrackLinePlot = Instance(DataView, transient=True)
    TrackPlotContainer = Instance(VPlotContainer, transient=True)

    DriftData = Instance(ArrayPlotData, transient=True)
    DriftPlot = Instance(DataView, transient=True)

    # scan image history
    history = Trait(utility.History)

    def _history_default(self):
        history = utility.History(length=10)
        history.put(self.copy_items(['X', 'Y', 'image', 'z', 'resolution']))
        #        history.put( ['X':self.X.copy(), 'Y':self.Y.copy(), 'image':self.image.copy(), 'z':self.z, 'resolution':self.resolution } )
        return history

    history_back = Button(label='Back')
    history_forward = Button(label='Forward')

    scan_range_reset = Button(label='Reset zoom')

    bidirectional = Bool(True)

    SetTrackPoint = Button(label='Set Track Point')
    AddTargetPoint = Button(label='Add Target', desc='Append current position to List of Targets')
    RemoveTargetPoint = Button(label='Remove Target', desc='Remove Last Target from List of Targets')

    # states and threads
    state = Enum('idle', 'scan', 'refocus_crosshair', 'refocus_trackpoint', 'xyz_monitor_scan')
    counter_state = Enum('idle', 'count')
    periodic_refocus = Bool(False, label='Periodic refocusing')
    thread_join_timeout = 10.

    def _state_changed(self):
        daemon_name = str(self.DaemonLoop)
        for k in self.threads.keys():
            if k != daemon_name:
                self.stop_thread(k)
        if self.state != 'idle':
            method = self.state2method[self.state]
            self.start_thread(method)

    def _counter_state_changed(self):
        self.stop_count_thread()
        if self.counter_state == 'count':
            self.count_thread = threading.Thread(target=self.count, name='confocal.count_thread' + utility.get_timestamp_string())
            self.count_thread.stop_request = threading.Event()
            self.count_thread.start()

    def stop_count_thread(self):
        if isinstance(self.count_thread, threading.Thread):
            if self.count_thread is None or self.count_thread is threading.current_thread() or not self.count_thread.isAlive():
                return
            self.count_thread.stop_request.set()
            self.count_thread.join(10.)
            self.count_thread = None

    def start_thread(self, method):
        name = str(method)
        pi3d.logger.debug('starting thread ' + name)
        self.threads[name] = threading.Thread(target=method, name='confocal.threads["' + name + '"] ' + utility.get_timestamp_string())
        self.threads[name].stop_request = threading.Event()
        self.threads[name].start()

    def stop_thread(self, method):
        name = str(method)
        pi3d.logger.debug('attempt to stop thread ' + name)
        if not self.threads.has_key(name):
            self.threads[name] = None
        th = self.threads[name]
        if th is None \
                or th is threading.current_thread() \
                or not th.isAlive():
            return
        pi3d.logger.debug('stopping thread ' + name)
        th.stop_request.set()
        th.join(10.)
        self.threads[name] = None

    def DaemonLoop(self):
        while True:
            try:
                threading.current_thread().stop_request.wait(self.TrackInterval)
                if threading.current_thread().stop_request.is_set():
                    break
                self.state = 'refocus_trackpoint'
            except Exception:
                pi3d.logger.exception(str(Exception))

    def _periodic_refocus_changed(self):
        self.stop_thread(self.DaemonLoop)
        if self.periodic_refocus:
            self.start_thread(self.DaemonLoop)

    def _X_default(self):
        return numpy.linspace(pi3d.nidaq._scanner_xrange[0], pi3d.nidaq._scanner_xrange[-1], self.resolution + 1)

    def _Y_default(self):
        return numpy.linspace(pi3d.nidaq._scanner_yrange[0], pi3d.nidaq._scanner_yrange[-1], self.resolution + 1)

    def _image_default(self):
        return numpy.zeros((len(self.X), len(self.Y)))

    def _x_changed(self):
        if self.TiltCorrection:
            pi3d.nidaq.scanner_set_pos(self.x, self.y, self.z + self._calc_dz(self.x, self.y))
        else:
            pi3d.nidaq.scanner_setx(self.x)

    def _y_changed(self):
        if self.TiltCorrection:
            pi3d.nidaq.scanner_set_pos(self.x, self.y, self.z + self._calc_dz(self.x, self.y))
        else:
            pi3d.nidaq.scanner_sety(self.y)

    def _z_changed(self):
        if self.state != 'scan':
            if self.TiltCorrection:
                pi3d.nidaq.scanner_set_pos(self.x, self.y, self.z + self._calc_dz(self.x, self.y))
            else:
                pi3d.nidaq.scanner_setz(self.z)

    def _calc_dz(self, x, y):
        return -((x - self.x0) * self.ax + (y - self.y0) * self.ay)

    def _SetR1Button_changed(self):
        self.r1 = numpy.array((self.x, self.y, self.z))

    def _SetR2Button_changed(self):
        self.r2 = numpy.array((self.x, self.y, self.z))

    def _SetR3Button_changed(self):
        self.r3 = numpy.array((self.x, self.y, self.z))

    @on_trait_change('TiltAngleButton')
    def SetTiltAngle(self):
        a = self.r2 - self.r1
        b = self.r3 - self.r1
        n = numpy.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))
        self.ax = n[0] / n[2]
        self.ay = n[1] / n[2]

    @on_trait_change('TiltReferenceButton')
    def SetTiltReference(self):
        self.x0 = self.x
        self.y0 = self.y

    def _C_default(self):
        return numpy.zeros((self.TraceLength,))

    def _T_default(self):
        return self.CountInterval * numpy.arange(self.TraceLength)

    def _T_changed(self):
        self.TraceData.set_data('t', self.T)

    def _C_changed(self):
        self.TraceData.set_data('y', self.C / 1000)
        # if self.trace_update_count % self.UpdateInterval == 0:
        #    self.TraceData.set_data('y', self.C/1000)
        #    self.trace_update_count = 1
        # else:
        #    self.trace_update_count += 1

    def _TraceLength_changed(self):
        self.C = self._C_default()
        self.T = self._T_default()
        if self.counter_state == 'count':
            self.init_counter()

    def _CountInterval_changed(self):
        self.C = self._C_default()
        self.T = self._T_default()
        if self.counter_state == 'count':
            self.init_counter()

    def _SetTrackPoint_changed(self):
        self.TrackPoint = numpy.array(((self.x, self.y, self.z),))
        self.drift = numpy.array(((0, 0, 0),))

    def _AddTargetPoint_changed(self):
        self.AddTarget([self.x, self.y, self.z])

    def _RemoveTargetPoint_changed(self):
        self.RemoveTarget(self.CurrentTargetIndex)
        self._CurrentTargetIndex_changed()

    def _ShowTargets_changed(self):
        self.TargetPlot.request_redraw()

    def _ScanData_default(self):
        return ArrayPlotData(image=self.image, x=numpy.array(()), y=numpy.array(()))

    def _ScanPlot_default(self):
        return Plot(self.ScanData, width=500, height=500, resizable='hv', aspect_ratio=1.0,
                    padding_top=10, padding_left=30, padding_right=10, padding_bottom=20)

    def _TargetPlot_default(self):
        return self.ScanPlot.plot(('x', 'y'), type='scatter', marker='cross', marker_size=6, line_width=1.0, color='black')[0]

    def _ScanImage_default(self):
        return self.ScanPlot.img_plot('image', colormap=jet,
                                      xbounds=(self.X[0], self.X[-1]), ybounds=(self.Y[0], self.Y[-1]))[0]

    def _cursor_default(self):
        cursor = CursorTool(self.ScanImage,
                            drag_button='left',
                            color='white',
                            line_width=1.0)
        cursor._set_current_position('x', (self.x, self.y))
        return cursor

    def _zoom_tool_default(self):
        return ZoomTool(self.ScanImage, enable_wheel=False)

    def _ScanPlotContainer_default(self):
        ScanImage = self.ScanImage
        ScanImage.x_mapper.domain_limits = (pi3d.nidaq._scanner_xrange[0], pi3d.nidaq._scanner_xrange[1])
        ScanImage.y_mapper.domain_limits = (pi3d.nidaq._scanner_yrange[0], pi3d.nidaq._scanner_yrange[1])
        ScanImage.overlays.append(self.zoom_tool)
        ScanImage.overlays.append(self.cursor)
        colormap = ScanImage.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=self.ScanPlot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            height=400,
                            padding_top=10, padding_bottom=20, padding_left=60, padding_right=10)

        container = HPlotContainer()
        container.add(self.ScanPlot)
        container.add(colorbar)
        container.spacing = 0

        return container

    def run_refocus(self):
        self.state = 'refocus_crosshair'
        while self.state == 'refocus_crosshair':
            time.sleep(0.1)

    def _TraceData_default(self):
        return ArrayPlotData(t=self.T, y=self.C)

    def _TracePlot_default(self):
        plot = Plot(self.TraceData,
                    padding_top=10, padding_left=10, padding_right=70, padding_bottom=20)
        plot.plot(('t', 'y'), type='line', color='white', bgcolor='black', line_width=2)
        plot.value_axis.title = 'kcounts / s'
        plot.y_axis.orientation = 'right'
        plot.x_grid = None
        plot.y_grid = None
        return plot

    def _TrackImageData_default(self):
        return ArrayPlotData(image=numpy.zeros((2, 2)))

    def _TrackImagePlot_default(self):
        plot = Plot(self.TrackImageData, width=100, height=100, padding_top=10, padding_right=10,
                    padding_left=65, padding_bottom=30)
        plot.index_mapper.domain_limits = (pi3d.nidaq._scanner_xrange[0], pi3d.nidaq._scanner_xrange[1])
        plot.value_mapper.domain_limits = (pi3d.nidaq._scanner_yrange[0], pi3d.nidaq._scanner_yrange[1])
        return plot

    def _TrackImage_default(self):
        return self.TrackImagePlot.img_plot('image', colormap=jet)[0]

    def _TrackLineData_default(self):
        return ArrayPlotData(t=self.ZT, y=self.zdata)

    def _TrackLinePlot_default(self):
        plot = Plot(self.TrackLineData, width=100, height=100, padding_top=0, padding_right=10,
                    padding_left=65, padding_bottom=40)
        plot.index_axis.title = 'z [micron]'
        plot.value_axis.title = 'kcounts / s'
        return plot

    def _TrackLine_default(self):
        return self.TrackLinePlot.plot(('t', 'y'), color='blue')[0]

    def _TrackPlotContainer_default(self):
        container = VPlotContainer(self.TrackLinePlot, self.TrackImagePlot)
        container.spacing = 0
        container.bgcolor = 'sys_window'
        return container

    def _DriftData_default(self):
        return ArrayPlotData(t=numpy.arange(self.drift.shape[0]), x=self.drift[:, 0], y=self.drift[:, 1], z=self.drift[:, 2])

    def _DriftPlot_default(self):
        plot = Plot(self.DriftData)
        plot.plot(('t', 'x'), type='line', color='blue')
        plot.plot(('t', 'y'), type='line', color='red')
        plot.plot(('t', 'z'), type='line', color='green')
        plot.index_axis.title = 'index'
        plot.value_axis.title = 'drift [micron]'
        return plot


        # automatic update of plot data

    def _image_changed(self):

        self.ScanData.set_data('image', self.image)
        self.ScanPlot.request_redraw()

    def reconstruct_confocal(self):
        self.X = self._debbug[0][1][0]
        self.X = numpy.append(self.X, self.X[-1])
        self.Y = numpy.empty(len(self._debbug[:-1]))
        self.X_monitor = numpy.empty((len(self._debbug[0][0][0]), len(self._debbug[:-1])))
        self.Y_monitor = numpy.empty((len(self._debbug[0][0][1]), len(self._debbug[:-1])))
        self.Z_monitor = numpy.empty((len(self._debbug[0][0][2]), len(self._debbug[:-1])))
        self.image = numpy.empty((len(self._debbug[0][0][2]), len(self._debbug[:-1])))
        for i in range(len(self._debbug[:-1])):
            self.Y[i] = self._debbug[i][1][1][0]
            if i % 2 == 0:
                self.X_monitor[i, :] = self._debbug[i][0][0]
                self.Y_monitor[i, :] = self._debbug[i][0][1]
                self.Z_monitor[i, :] = self._debbug[i][0][2]
                self.image[i, :] = self._debbug[i][2]
            else:
                self.X_monitor[i, :] = self._debbug[i][0][0][::-1]
                self.Y_monitor[i, :] = self._debbug[i][0][1][::-1]
                self.Z_monitor[i, :] = self._debbug[i][0][2][::-1]
                self.image[i, :] = self._debbug[i][2][::-1]
        self.Y = numpy.append(self.Y, self.Y[-1])
        self.interpolate = True
        self._image_changed()

    def _xydata_changed(self):
        self.TrackImageData.set_data('image', self.xydata)
        self.TrackImagePlot.request_redraw()

    def _zdata_changed(self):
        self.TrackLineData.set_data('y', self.zdata)

    def _ZT_changed(self):
        self.TrackLineData.set_data('t', self.ZT)

    def _zfit_data_changed(self):
        plot = self.TrackLinePlot
        while len(plot.components) > 1:
            plot.remove(plot.components[-1])
        self.TrackLineData.set_data('fit', self.zfit_data)
        plot.plot(('t', 'fit'), style='line', color='red')
        plot.request_redraw()

    def _drift_changed(self):
        self.DriftData.set_data('x', self.drift[:, 0])
        self.DriftData.set_data('y', self.drift[:, 1])
        self.DriftData.set_data('z', self.drift[:, 2])
        self.DriftData.set_data('t', numpy.arange(self.drift.shape[0]))

    def _TargetList_changed(self):
        if len(self.TargetList) == 0:
            self.ScanData.set_data('x', numpy.array(()))
            self.ScanData.set_data('y', numpy.array(()))
        else:
            positions = numpy.array(self.TargetList)
            self.ScanData.set_data('x', positions[:, 0])
            self.ScanData.set_data('y', positions[:, 1])

    def init_counter(self):
        self.C = self._C_default()
        pi3d.timetagger.init_counter('counter', interval_ps=int(self.CountInterval * 1e12), trace_length=self.TraceLength)
        pi3d.timetagger.counter.clear()

    def ReadoutInterval_changed(self):
        print('Readoutinterval has been changed')
        if self.counter_state == 'count':
            self.init_counter()

    def count(self):
        """Acquire Count Trace"""
        self.init_counter()
        while True:
            if threading.current_thread().stop_request.is_set():
                break
            self.C = numpy.array(pi3d.timetagger.counter.getData()[0] / self.CountInterval)
            time.sleep(self.ReadoutInterval)
        pi3d.timetagger.counter.clear()

    def scan(self):
        """Acquire x-y-scan"""
        if pi3d.nidaq.init_scan(self.SettlingTime, self.CountTime) != 0:  # returns 0 if successful
            print('error in nidaq')
            return

        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2

        if (x2 - x1) >= (y2 - y1):
            X = numpy.linspace(x1, x2, self.resolution)
            Y = numpy.linspace(y1, y2, int(self.resolution * (y2 - y1) / (x2 - x1)))
        else:
            Y = numpy.linspace(y1, y2, self.resolution)
            X = numpy.linspace(x1, x2, int(self.resolution * (x2 - x1) / (y2 - y1)))

        self.X = X
        self.Y = Y

        XP = X[::-1]

        self.image = numpy.zeros((len(Y), len(X)))
        self.ScanImage.index.set_data(X, Y)

        for i, y in enumerate(Y):
            if threading.current_thread().stop_request.is_set():
                break
            if i % 2 == 0 and self.bidirectional:
                XL = X
            else:
                XL = XP
            YL = y * numpy.ones(X.shape)
            ZL = self.z * numpy.ones(X.shape)
            if self.TiltCorrection:
                ZL = self.z + self._calc_dz(XL, YL)
            else:
                ZL = self.z * numpy.ones(X.shape)
            Line = numpy.vstack((XL, YL, ZL))
            c = pi3d.nidaq.scan_line(Line) / 1e3
            if i % 2 == 0 and self.bidirectional:
                if self.scan_offset != 0:
                    self.image[i, :-self.scan_offset] = c[self.scan_offset:]
                else:
                    self.image[i, :] = c[:]
            elif self.bidirectional:
                if self.scan_offset != 0:
                    self.image[i, self.scan_offset:] = c[self.scan_offset:][::-1]
                else:
                    self.image[i, :] = c[::-1]
            else:
                self.image[i, :] = c[::-1]
            self._image_changed()

        pi3d.nidaq.stop_scan()

        if self.x < x1 or self.x > x2 or self.y < y1 or self.y > y2:
            self.x = (x1 + x2) / 2
            self.y = (y1 + y2) / 2
        else:
            pi3d.nidaq.scanner_set_pos(self.x, self.y, self.z)

        # save scan data to history
        self.history.put(self.copy_items(['X', 'Y', 'image', 'z', 'resolution']))
        #        self.history.put( {'X':X.copy(), 'Y':Y.copy(), 'image':self.image.copy(), 'z':self.z, 'resolution':self.resolution } )
        #                           'targets':self.TargetList[:],
        #                           'TrackPoint':self.TrackPoint, 'drift':self.drift[-1] } )

        if not threading.current_thread().stop_request.is_set():
            self.state = 'idle'

    def xyz_monitor_scan(self):
        """Acquire x-y-scan and use xyz monitor signal for exact confocal scan image"""
        if self.CrossTalkCorrectionMonitor:
            if not hasattr(self, 'cross_talk_data_container'):
                self.start_measure_cross_talk()
            cross_talk_shift = self.cross_talk_data_container.get_cross_talk_shift()
        else:
            cross_talk_shift = None

        confocal_scan = ConfocalLineScan(
            CountTime=self.CountTimeMonitor,
            SettlingTime=0.00001,
            cross_talk_shift=cross_talk_shift
        )
        post_samples = 59

        # Activate for Simulation
        # confocal_scan = SimulateConfocalLineScanScan2dClass('D:\\python\\Confocal_solid_magnet\\extreme_scan_class_extens.pyd')
        # confocal_scan = SimulateConfocalLineScan('D:\\python\\Confocal_3TMagnet\\_debbug.pyd')
        # self.x1 , self.x2, self.y1, self.y2, self.resolution, self.bidirectional, post_samples = confocal_scan.get_measurement_parameter()

        if self.TiltCorrection:
            calc_dz = self._calc_dz
        else:
            calc_dz = None

        self.scan_data_container = Scan2d(
            x1=self.x1,
            x2=self.x2,
            y1=self.y1,
            y2=self.y2,
            plane_z=self.z,
            post_samples=post_samples,
            resolution=self.resolution,
            calc_dz=calc_dz,
            TimeDelay=self.ReadoutDelayMonitor,
            SettlingTime=0.00001,
            CountTime=self.CountTimeMonitor,
            bidirectional=self.bidirectional,
            interpolate=True
        )

        self.X = self.scan_data_container.X_grid()
        self.Y = self.scan_data_container.Y_grid()

        self.ScanImage.index.set_data(self.scan_data_container.X_grid(), self.scan_data_container.Y_grid())

        for i, Line in self.scan_data_container.get_target_data():
            if threading.current_thread().stop_request.is_set():
                self.image = self.scan_data_container.get_full_interpolation()
                break

            print('Line: ', i)

            self.scan_data_container.append(confocal_scan(Line))

            self.image = self.scan_data_container.get_interpolation()
            # self._image_changed()

        del confocal_scan

        self.image = self.scan_data_container.get_full_interpolation()
        # self._image_changed()

        if self.scan_data_container.target_cross_lost(self.x, self.y):
            self.x, self.y = self.scan_data_container.get_new_target_cross()
        else:
            pi3d.nidaq.scanner_set_pos(self.x, self.y, self.z)

        # save scan data to history
        self.history.put(self.copy_items(['X', 'Y', 'image', 'z', 'resolution', 'scan_data_container']))
        # self.history.put( {'X':X.copy(), 'Y':Y.copy(), 'image':self.image.copy(), 'z':self.z, 'resolution':self.resolution } )
        #                    'targets':self.TargetList[:],
        #                    'TrackPoint':self.TrackPoint, 'drift':self.drift[-1] } )

        if not threading.current_thread().stop_request.is_set():
            self.state = 'idle'

    def load_scan(self, file_name):
        # NOT USED
        # loading old measuered data through an scan_data_container to Confocal
        # scan_data_container is used by xyz_monitor_scan
        self.scan_data_container = LoadScan2d(file_name)

        self.X = self.scan_data_container.X_grid()
        self.Y = self.scan_data_container.Y_grid()

        self.image = self.scan_data_container.get_full_interpolation()

    def calc_line_displacement(self, line1, line2, pixel_min, pixel_max, pixel_step):
        # Used by measure_displacement
        # Calculates how many pixels two lines, line1 and line2, are displaced from each other (lowest of quadratic error sum)
        # pixel_min and pixel_max: How many pixels minimal respectivily maximal, the algorithm displaces data lines while calculation
        # pixel_step: pixel distance between two calculation points
        from scipy import interp
        first_enter = True
        lowest_difference = 0
        lowest_difference_pixel = 0

        if len(line1) == len(line2):
            pixels_count = len(line1)
        else:
            print('error: in confocal.calc_line_displacement array length different')

        pixels = numpy.arange(0, pixels_count, 1)

        if pixel_max > 0:
            match_range_high = numpy.ceil(numpy.abs(pixel_max))
        else:
            match_range_high = 0

        if pixel_min < 0:
            match_range_low = numpy.ceil(numpy.abs(pixel_min))
        else:
            match_range_low = 0

        low_index = match_range_low
        high_index = pixels_count - match_range_high

        pixels_in_range = numpy.arange(low_index, high_index)
        array2 = line2[low_index:high_index]

        for pixel in numpy.arange(pixel_min, pixel_max, pixel_step):
            array1 = interp(pixels_in_range, pixels - pixel, line1)

            temp_difference = ((array2 - array1) ** 2).sum()
            # print array1,array2

            if first_enter:
                first_enter = False
                lowest_difference = temp_difference
                lowest_difference_pixel = pixel
            elif lowest_difference > temp_difference:
                lowest_difference = temp_difference
                lowest_difference_pixel = pixel

        return lowest_difference_pixel

    def measure_displacement(self, y_postion, repeat, pixel_min, pixel_max, pixel_resolution):
        # Not in use
        # Measures pixel displacement between left to right scan and right to left scan, in x-direction
        # y_position: defines y-position while measurement
        # repeat: Number of repetitions of that measurement
        # pixel_min, pixel_max, pixel_resolution: passing parameter for calc_line_displacement (see there for more information)
        """Acquire x-y-scan"""
        SettlingTime = 0.00001
        CountTime = self.CountTime
        if pi3d.nidaq.init_scan(SettlingTime, CountTime) != 0:  # returns 0 if successful
            print('error in nidaq')
            return

        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2

        if (x2 - x1) >= (y2 - y1):
            X = numpy.linspace(x1, x2, self.resolution)
        else:
            X = numpy.linspace(x1, x2, int(self.resolution * (x2 - x1) / (y2 - y1)))

        Y = numpy.array(2 * repeat * [y_postion])

        XP = X[::-1]

        min_offset = []
        max_offset = []

        lines = []

        for i, y in enumerate(Y):
            if i % 2 == 0:
                XL = X
            else:
                XL = XP
            YL = y * numpy.ones(X.shape)
            ZL = self.z * numpy.ones(X.shape)
            if self.TiltCorrection:
                ZL = self.z + self._calc_dz(XL, YL)
            else:
                ZL = self.z * numpy.ones(X.shape)
            Line = numpy.vstack((XL, YL, ZL))

            xyz_scan_position, counts = pi3d.nidaq.scan_read_line(Line)
            counts = counts / 1e3

            # interpolate_counts = interp1d(xyz_scan_position[0], counts, kind='cubic')
            # c = interpolate_counts(XL)
            from scipy import interp
            time = numpy.arange(len(Line[0])) * (SettlingTime + CountTime)
            if i % 2 == 0:
                xyz_scan_position[0] = interp(time, time, xyz_scan_position[0])
                c = interp(XL, xyz_scan_position[0][:], counts[:])
            else:
                xyz_scan_position[0] = interp(time, time, xyz_scan_position[0])
                c = interp(XL, xyz_scan_position[0][::-1], counts[::-1])

            min_offset.append(min(xyz_scan_position[0]))
            max_offset.append(max(xyz_scan_position[0]))

            if i % 2 == 0:
                lines.append([c])
            else:
                lines[-1].append(c[::-1])

        lines

        pixel_array = numpy.array([])
        for i, line in enumerate(lines):
            pixel_array = numpy.append(pixel_array, self.calc_line_displacement(line[0], line[1], pixel_min, pixel_max, pixel_resolution))

        pi3d.nidaq.stop_scan()

        if self.x < x1 or self.x > x2 or self.y < y1 or self.y > y2:
            self.x = (x1 + x2) / 2
            self.y = (y1 + y2) / 2
        else:
            pi3d.nidaq.scanner_set_pos(self.x, self.y, self.z)

            # save scan data to history
            # self.history.put( self.copy_items(['X', 'Y', 'image', 'z', 'resolution'] ) )
        #        self.history.put( {'X':X.copy(), 'Y':Y.copy(), 'image':self.image.copy(), 'z':self.z, 'resolution':self.resolution } )
        #                           'targets':self.TargetList[:],
        #                           'TrackPoint':self.TrackPoint, 'drift':self.drift[-1] } )

        return pixel_array.mean()

    def goto(self, position):
        pi3d.nidaq.scanner_set_pos(position[0], position[1], position[2])

    def _MeasureCrossTalkMonitor_changed(self):
        if self.start_measure_cross_talk.__repr__() in self.threads.keys() and not (self.threads[self.start_measure_cross_talk.__repr__()] == None):
            self.stop_thread(self.start_measure_cross_talk.__repr__())
            self.MeasureCrossTalkLabelMonitor = '(Re)Measure cross talk'
        else:
            self.start_thread(self.start_measure_cross_talk)
            self.MeasureCrossTalkLabelMonitor = 'Abort measurement'

    def start_measure_cross_talk(self):
        self.measure_cross_talk(self.MeasureCrossTalkVelocityMonitor / 100, self.MeasureCrossTalkVelocityMonitor / 100, self.MeasureCrossTalkResolutionMonitor)

    def measure_cross_talk(self, start_velocity, stop_velocity, resolution):
        # Not Debugged, probably bad resoults for higher velocities
        # Measures cross talk between the 3 positioning scanner lines
        # start_velocity,stop_velocity: defines the absoulute minimum respectivily maximum velocity, measurement is done.
        # resolution: How many measurements are taken in between start and stop_velocity inclusive start and stop_velocity
        # Caution: this function repeats measurement for positive velocities and negative velocities areas and all 3 dimension.
        # so for example to measure on all three lines from -10 000 to +10 000 [micrometer/s] with 201 steps enter measure_cross_talk(100,10000,100)
        # that gives back a cross talk matrix object that can be called with an velocity vector like: array([[vx],[vy],[vz]])
        # crosstalk matrix will give back deltaposition = array([[delatx],[deltay],[deltaz]])
        self.cross_talk_data_container = MeasureCrossTalk(
            start_velocity,
            stop_velocity,
            resolution
        )

        for dimension, positive_direction, velocity, data in self.cross_talk_data_container.get_target_data():
            SettlingTime, CountTime = self.cross_talk_data_container.get_scan_parameter()
            confocal_scan = ConfocalLineScan(
                CountTime=CountTime,
                SettlingTime=SettlingTime,
            )
            self.cross_talk_data_container.append(confocal_scan(data))
            del confocal_scan
            if threading.current_thread().stop_request.is_set():
                break

        return self.cross_talk_data_container.get_cross_talk_shift()

    def old_measure_cross_talk(self, start_velocity, stop_velocity, resolution):
        # Not used
        # same as measure_cross_talk
        low_position = numpy.array([[pi3d.nidaq._scanner_xrange[0]], [pi3d.nidaq._scanner_yrange[0]], [pi3d.nidaq._scanner_zrange[0]]])
        high_position = numpy.array([[pi3d.nidaq._scanner_xrange[1]], [pi3d.nidaq._scanner_yrange[1]], [pi3d.nidaq._scanner_zrange[1]]])
        dimensions = len(low_position)
        start_vector = numpy.array([[0], [1. / 2], [1. / 2]])
        end_vector = numpy.array([[1], [1. / 2], [1. / 2]])
        cross_talk_shift = []

        def zero(v):
            return 0

        import time

        pos_resolution = 500

        self._debbug = []

        for dimension in range(dimensions):
            velocities = numpy.array([])
            dimension_plus_1 = (dimension + 1) % (len(low_position))
            dimension_plus_2 = (dimension + 2) % (len(low_position))
            crossing1 = numpy.array([])
            crossing2 = numpy.array([])
            for positive_direction in [False, True]:
                for velocity in numpy.linspace(start_velocity, stop_velocity, resolution):
                    SettlingTime = 0.00001
                    CountTime = numpy.abs((high_position[dimension][0] - low_position[dimension][0]) / (velocity * ((high_position[dimension][0] - low_position[dimension][0]) / (high_position[0][0] - low_position[0][0])) * pos_resolution)) - SettlingTime
                    print(CountTime, high_position[dimension], low_position[dimension], velocity, SettlingTime)
                    if pi3d.nidaq.init_scan(SettlingTime, CountTime) != 0:  # returns 0 if successful
                        print('error in nidaq')
                        return
                    start_point = high_position * numpy.roll(start_vector, dimension)
                    stop_point = high_position * numpy.roll(end_vector, dimension)

                    if not positive_direction:
                        velocities = numpy.append(velocities, -velocity)
                        # self.x = stop_point[0][0]
                        # self.y = stop_point[1][0]
                        # self.z = stop_point[2][0]
                        print(stop_point[0][0])
                        print(stop_point[1][0])
                        print(stop_point[2][0])
                        # pi3d.nidaq.scanner_set_pos(stop_point[0][0],stop_point[1][0],stop_point[2][0])
                        pi3d.nidaq.scan_read_line(line_array(stop_point, stop_point, 2, 0, 0, positive_direction))
                    else:
                        print(start_point[0][0])
                        print(start_point[1][0])
                        print(start_point[2][0])
                        velocities = numpy.append(velocities, velocity)
                        # pi3d.nidaq.scanner_set_pos(start_point[0][0],start_point[1][0],start_point[2][0])
                        pi3d.nidaq.scan_read_line(line_array(start_point, start_point, 2, 0, 0, positive_direction))

                    time.sleep(0.1)
                    points, forget = pi3d.nidaq.scan_read_line(line_array(start_point, stop_point, pos_resolution, 0, 0, positive_direction))
                    self._debbug.append((points, line_array(start_point, stop_point, pos_resolution, 0, 0, positive_direction)))
                    # print stop_point[dimension_plus_2][-1]
                    # print stop_point[dimension_plus_2]
                    # print dimension_plus_2
                    # print points
                    # print points[dimension_plus_2]
                    # print points[dimension_plus_2][-1]

                    # print dimension_plus_2

                    reference = []
                    if positive_direction:
                        reference.append(stop_point[dimension_plus_1][-1])
                        reference.append(stop_point[dimension_plus_2][-1])
                    else:
                        reference.append(start_point[dimension_plus_1][-1])
                        reference.append(start_point[dimension_plus_2][-1])

                    print(reference)
                    print(start_point[dimension_plus_1][-1])
                    print(stop_point[dimension_plus_1][-1])
                    print(start_point[dimension_plus_2][-1])
                    print(stop_point[dimension_plus_2][-1])
                    print(points[dimension_plus_1][-1])
                    print(points[dimension_plus_2][-1])

                    crossing1 = numpy.append(crossing1, reference[0] - points[dimension_plus_1][-1])
                    crossing2 = numpy.append(crossing2, reference[1] - points[dimension_plus_2][-1])
                if not positive_direction:
                    velocities = velocities[::-1]
                    crossing1 = crossing1[::-1]
                    crossing2 = crossing2[::-1]
                    velocities = numpy.append(velocities, 0)
                    crossing1 = numpy.append(crossing1, 0)
                    crossing2 = numpy.append(crossing2, 0)
            cross_talk_shift_line = [zero, interp_orig_lin(velocities, crossing1), interp_orig_lin(velocities, crossing2)]
            cross_talk_shift.append(cross_talk_shift_line[-dimension:] + cross_talk_shift_line[:-dimension])

        self.cross_talk_shift = cross_talk_matrix(cross_talk_shift)
        return self.cross_talk_shift

    @cached_property
    def _get_CurrentTarget(self):
        if self.CurrentTargetIndex == -1 or len(self.TargetList) == 0:
            return (0, 0)
        else:
            return self.TargetList[self.CurrentTargetIndex]

    def _CurrentTargetIndex_changed(self):
        if self.CurrentTargetIndex == -1 or len(self.TargetList) == 0:
            return
        elif self.CurrentTargetIndex >= len(self.TargetList):
            self.CurrentTargetIndex = len(self.TargetList) - 1
        self.x = self.TargetList[self.CurrentTargetIndex][0]
        self.y = self.TargetList[self.CurrentTargetIndex][1]
        self.z = self.TargetList[self.CurrentTargetIndex][2]

    def _next_target_btn_changed(self):
        self.CurrentTargetIndex = (self.CurrentTargetIndex + 1) % len(self.TargetList)

    def _previous_target_btn_changed(self):
        self.CurrentTargetIndex = (self.CurrentTargetIndex - 1) % len(self.TargetList)

    def _remove_duplicate_targets_changed(self, dist_x_y=1, dist_z=2):
        remove = []
        for t1 in range(len(self.TargetList)):
            for t2 in range(t1 + 1, len(self.TargetList)):
                dxy = ((self.TargetList[t1][0] - self.TargetList[t2][0]) ** 2 + (self.TargetList[t1][1] - self.TargetList[t2][1]) ** 2) ** 0.5
                dz = abs(self.TargetList[t1][2] - self.TargetList[t2][2])
                if dxy < dist_x_y and dz < dist_z:
                    remove.append(t2)
        remove = numpy.sort(remove)
        for i in range(len(remove)):
            self.TargetList.pop(remove[-i - 1])

    def AddTarget(self, position, index=None):
        if index is None:
            self.TargetList.append(position)
        else:
            self.TargetList.insert(index, position)
        self._TargetList_changed()

    def RemoveTarget(self, index=None):
        if index is None:
            self.TargetList.pop()
        else:
            self.TargetList.pop(index)
        self._TargetList_changed()

    def track(self):
        print("Seems like you misclicked. I fixed that for you and did nothing.")
        self.state = 'idle'
        if False:
            manager = utility.TrackerManagerSingleton()
            manager.request_pause()
            pi3d.logger.debug('got the pause. starting...')

            if len(self.TrackPoint) == 0:
                self.TrackPoint = numpy.array(((self.x, self.y, self.z),))

            self.x, self.y, self.z = self.TrackPoint[-1]
            self.TrackPoint = numpy.append(self.TrackPoint, (self.refocus(),), axis=0)
            drift = self.TrackPoint[-1] - self.TrackPoint[0]
            self.drift = numpy.append(self.drift, (drift,), axis=0)
            logging.getLogger().info('track: drift=%.2f, %.2f, %.2f' % tuple(drift))

            utility.SaveFigure(self.TrackImagePlot, pi3d.log_dir + 'tracker/TrackImage' + utility.get_timestamp_string() + '.png')

            if self.TargetMode:
                self.x, self.y, self.z = self.CurrentTarget
                logging.getLogger().info('track: moved to target')
                self.refocus()
            pi3d.logger.debug('done. releasing pause...')
            manager.release_pause()
            if not threading.current_thread().stop_request.is_set():
                self.state = 'idle'

    def refocus(self):
        """Refocuses around current position an x, y, and z-direction
        """
        pi3d.md.stop_awgs()
        pi3d.mcas_dict['green'].initialize(trigger_mode='continuous')
        pi3d.mcas_dict['green'].start_awgs()
        time.sleep(0.4)
        try:
            if pi3d.nidaq.init_scan(self.XYSettlingTime, self.XYCountTime) != 0:  # returns 0 if successful
                print('error in nidaq')
                return
            xp = x0 = self.x - self.XOffset
            yp = y0 = self.y - self.YOffset
            zp = z0 = self.z
            ##+pi3d.nidaq._scanner_xrange[1]
            safety = 0  # distance to keep from the ends of scan range
            xmin = numpy.clip(x0 - 0.5 * self.XYSize, pi3d.nidaq._scanner_xrange[0] + safety, pi3d.nidaq._scanner_xrange[1] - safety)
            xmax = numpy.clip(x0 + 0.5 * self.XYSize, pi3d.nidaq._scanner_xrange[0] + safety, pi3d.nidaq._scanner_xrange[1] - safety)
            ymin = numpy.clip(y0 - 0.5 * self.XYSize, pi3d.nidaq._scanner_yrange[0] + safety, pi3d.nidaq._scanner_yrange[1] - safety)
            ymax = numpy.clip(y0 + 0.5 * self.XYSize, pi3d.nidaq._scanner_yrange[0] + safety, pi3d.nidaq._scanner_yrange[1] - safety)

            X = numpy.arange(xmin, xmax, self.XYStep)
            Y = numpy.arange(ymin, ymax, self.XYStep)

            self.XT = X
            self.YT = Y

            XP = X[::-1]

            self.xydata = numpy.zeros((len(Y), len(X)))
            self.TrackImage.index.set_data(X, Y)

            for i, y in enumerate(Y):
                if i % 2 == 0:
                    XL = X
                else:
                    XL = XP
                YL = y * numpy.ones(X.shape)
                if self.TiltCorrection:
                    ZL = z0 + self._calc_dz(XL, YL)
                else:
                    ZL = z0 * numpy.ones(X.shape)
                ZL = numpy.clip(ZL, pi3d.nidaq._scanner_zrange[0] + safety, pi3d.nidaq._scanner_zrange[1] - safety)
                Line = numpy.vstack((XL, YL, ZL))
                c = pi3d.nidaq.scan_line(Line) / 1e3
                if i % 2 == 0:
                    self.xydata[i, :] = c[:]
                else:
                    self.xydata[i, :] = c[::-1]
                self._xydata_changed()
            pi3d.nidaq.stop_scan()

            xp, yp = self.FitXY()

            self.x = xp + self.XOffset
            self.y = yp + self.YOffset

            # Z = numpy.hstack( ( numpy.arange(z0, z0-0.5*self.ZSize, -self.ZStep),
            #                    numpy.arange(z0-0.5*self.ZSize, z0+0.5*self.ZSize, self.ZStep),
            #                    numpy.arange(z0+0.5*self.ZSize, z0, -self.ZStep) ) )


            zsteps = int(0.5 * self.ZSize / self.ZStep)
            Z = numpy.hstack((numpy.arange(z0, z0 - (zsteps - 0.5) * self.ZStep, -self.ZStep),
                              numpy.arange(z0 - zsteps * self.ZStep, z0 + (zsteps - 0.5) * self.ZStep, self.ZStep),
                              numpy.arange(z0 + zsteps * self.ZStep, z0 + 0.5 * self.ZStep, -self.ZStep)))

            Z = numpy.clip(Z, pi3d.nidaq._scanner_zrange[0] + safety, pi3d.nidaq._scanner_zrange[1] - safety)

            X = xp * numpy.ones(Z.shape)
            Y = yp * numpy.ones(Z.shape)

            if not threading.current_thread().stop_request.is_set():
                pi3d.nidaq.init_scan(self.ZSettlingTime, self.ZCountTime)
                if self.TiltCorrection:
                    Line = numpy.vstack((X, Y, Z + self._calc_dz(xp, yp)))
                else:
                    Line = numpy.vstack((X, Y, Z))
                zdata = pi3d.nidaq.scan_line(Line) / 1e3
                pi3d.nidaq.stop_scan()

                self.ZT = Z
                self.zdata = zdata

                plot = self.TrackLinePlot
                while len(plot.components) > 1:
                    plot.remove(plot.components[-1])
                zp = self.FitZ()
            else:
                zp = z0

            pi3d.nidaq.stop_scan()
            pi3d.md.stop_awgs()
            self.z = zp
            self._z_changed()
            logging.getLogger().info('Refocus drift [nm]: %.3f , %.3f, %.3f' % ((self.x-x0)*1e3, (self.y - y0)*1e3, (self.z - z0)*1e3))
            if not threading.current_thread().stop_request.is_set():
                self.state = 'idle'
            # x, y, z = pi3d.nidaq.scanner_volt_to_pos(pi3d.nidaq.read_scanner_ai())[:,0]
            pi3d.save_values_to_file([self.x, self.y, self.z], 'confocal_pos')
            # pi3d.save_values_hdf(classifier='confocal_pos', vd=dict(x=self.x, y=self.y, z=self.z))
            pi3d.save_values_to_file([], 'confocal_pos_change')
            # pi3d.save_values_hdf(classifier='confocal_pos_change', vd=dict(x=(self.x-x0)*1e3, y=(self.y - y0)*1e3, z=(self.z - z0)*1e3))
        except Exception as e:
            logging.getLogger().exception(str(e))
        finally:
            return xp, yp, zp

    def FitXY(self):
        if self.XYFitMethod == 'Maximum':
            index = self.xydata.argmax()
            xp = self.XT[index % len(self.XT)]
            yp = self.YT[index * 1.0/len(self.XT)]
            self.XYFitParameters = [xp, yp]
            self.xfit = xp
            self.yfit = yp
            return xp, yp
        elif self.XYFitMethod == 'Gaussian':
            gaussfit = Fit.Gaussfit2D(self.xydata)
            params = gaussfit.Execute()
            gauss = gaussfit.gauss(*params)(*numpy.indices(self.xydata.shape))
            x_fit = params[3]
            y_fit = params[2]
            xp = self.XT[0] + (x_fit * self.XYStep)
            yp = self.YT[0] + (y_fit * self.XYStep)
            if abs(x_fit * self.XYStep) > self.XYSize or abs(y_fit * self.XYStep) > self.XYSize:
                # if fit-maximum out of refocus-range goto max datapoint
                index = self.xydata.argmax()
                xp = self.XT[int(index % len(self.XT))]
                yp = self.YT[int(index / len(self.XT))]
                self.XYFitParameters = [xp, yp]
                self.xfit = xp
                self.yfit = yp
                print(() % (xp, yp))
                return xp, yp
            # print "Refocus: x = %s, y = %s" %(xp, yp)
            return xp, yp
        elif self.XYFitMethod == 'ignore':
            xp = self.x - self.XOffset
            yp = self.y - self.YOffset
            return xp, yp
        else:
            print('Not Implemented! Fix Me!')

    def FitZ(self):
        if self.ZFitMethod == 'Maximum':
            zp = self.ZT[self.zdata.argmax()]
            self.zfit = zp
            return zp
        elif self.ZFitMethod == 'Polynomial(4th-order)':
            polyfit = numpy.polyfit(self.ZT, self.zdata, 4)
            # print polyfit
            poly = numpy.poly1d(polyfit)
            xfit = numpy.linspace(self.ZT.min(), self.ZT.max(), 100000)
            data_fit = numpy.zeros(xfit.shape)
            for i in range(len(xfit)):
                data_fit[i] = poly(xfit[i])
            fit = []
            for i in self.ZT:
                fit.append(poly(i))
            self.zfit_data = fit[:]
            self._zfit_data_changed()
            zp = xfit[data_fit.argmax()]
            self.zfit = zp
            return zp
        else:
            print('Not Implemented! Fix Me!')

    def FitZ_lmfit(self):
        if self.ZFitMethod == 'Maximum':
            zp = self.ZT[self.zdata.argmax()]
            self.zfit = zp
            return zp
        elif self.ZFitMethod == 'Polynomial(4th-order)':
            polyfit = numpy.polyfit(self.ZT, self.zdata, 4)
            # print polyfit
            poly = numpy.poly1d(polyfit)
            xfit = numpy.linspace(self.ZT.min(), self.ZT.max(), 100000)
            data_fit = numpy.zeros(xfit.shape)
            for i in range(len(xfit)):
                data_fit[i] = poly(xfit[i])
            fit = []
            for i in self.ZT:
                fit.append(poly(i))
            self.zfit_data = fit[:]
            self._zfit_data_changed()
            zp = xfit[data_fit.argmax()]
            self.zfit = zp
            return zp
        else:
            print('Not Implemented! Fix Me!')


    def _LowLimit_changed(self):
        self.ScanImage.value_range.low = self.LowLimit
        self.ScanImage.request_redraw()

    def _HighLimit_changed(self):
        self.ScanImage.value_range.high = self.HighLimit
        self.ScanImage.request_redraw()

    def _Autoscale_changed(self):
        self.ScanImage.value_range.high_setting = 'auto'
        self.ScanImage.value_range.low_setting = 'auto'
        self.ScanImage.request_redraw()

    def save_scan_plot(self, filename=None):
        if filename is None:
            filename = pi3d.get_filename() + '_confocal.png'
        else:
            filename = pi3d.get_filename(filename)
        self.save_figure(self.ScanPlotContainer, filename)

    def save_scan_data(self, filename=None, mode='bin'):
        if filename is None:
            filename = pi3d.get_filename() + '_confocal.pyd'
        else:
            filename = pi3d.get_filename(filename)
        self.dump_items(['X', 'Y', 'image', 'x', 'y', 'z', 'resolution', 'TrackPoint', 'TargetList'], filename=filename, mode=mode)

    def _history_back_fired(self):
        self.set_items(self.history.back())

    def _history_forward_fired(self):
        self.set_items(self.history.forward())

    def _scan_range_reset_fired(self):
        self.ScanPlot.index
        self.ScanImage.x_mapper.domain_limits = (pi3d.nidaq._scanner_xrange[0], pi3d.nidaq._scanner_xrange[1])
        self.ScanImage.y_mapper.domain_limits = (pi3d.nidaq._scanner_yrange[0], pi3d.nidaq._scanner_yrange[1])

    def _SaveImage_changed(self):
        bounds = self.ScanPlotContainer.outer_bounds
        gc = PlotGraphicsContext(bounds, dpi=72)
        gc.render_component(self.ScanPlotContainer)
        gc.save(self.FilePath + self.FileName + '.png')

    def reset_settings(self):
        self.counter_state = 'idle'
        self.SettlingTime = 0.0001
        self.CountTime = 0.002
        self.scan_offset = 0
        self.CountTImeMonitor = 0.002
        self.ReadoutDelayMonitor = 0.002
        self.CrossTalkCorrectionMonitor = False
        self.MeasureCrossTalkVelocityMonitor = 10000
        self.MeasureCrossTalkResolutionMonitor = 10
        self.XYSize = .25
        self.XYStep = 0.06
        self.XYSettlingTime = 0.01
        self.XYCountTime = 0.1
        self.XOffset = 0
        self.YOffset = 0
        self.ZSize = .5
        self.ZStep = 0.06
        self.ZSettlingTime = 0.01
        self.ZCountTime = 0.1
        self.XYFitMethod = 'Gaussian'
        self.ZFitMethod = 'Polynomial(4th-order)'

    # GUI
    MainView = View(
        VGroup(
            Tabbed(
                HGroup(
                    VGroup(
                        Item('ScanPlotContainer',
                             editor=ComponentEditor(),
                             show_label=False,
                             resizable=True,
                             enabled_when='state == "idle" or state == "count"'),
                        HGroup(Item('resolution', enabled_when='state == "idle" or state == "count"'),
                               Item('max_range_button'),
                               Item('x1'),
                               Item('x2'),
                               Item('y1'),
                               Item('y2'),
                               Item('TiltCorrection'),
                               Item('bidirectional'), ),
                        HGroup(Item('LowLimit'),
                               Item('HighLimit'),
                               Item('Autoscale', show_label=False),
                               Item('history_back', show_label=False),
                               Item('history_forward', show_label=False),
                               Item('ShowTargets'),
                               Item('FileName'),
                               Item('SaveImage', show_label=False)),
                        Item('x', enabled_when='state == "idle" or state == "count"'),
                        Item('y', enabled_when='state == "idle" or state == "count"'),
                        Item('z', enabled_when='state == "idle" or state == "count" or state == "scan"'), ),
                    label='Navigation'),
                VGroup(
                    HGroup(
                        Item('r1'),
                        Item('SetR1Button', show_label=False),
                    ),
                    HGroup(
                        Item('r2'),
                        Item('SetR2Button', show_label=False),
                    ),
                    HGroup(
                        Item('r3'),
                        Item('SetR3Button', show_label=False),
                    ),
                    HGroup(
                        Item('x0'),
                        Item('y0'),
                        Item('TiltReferenceButton', show_label=False)),
                    HGroup(
                        Item('TiltAngleButton', show_label=False)
                    ),
                    label='Tilt'),
                VGroup(
                    HGroup(
                        VGroup(
                            Item('SettlingTime'),
                            Item('CountTime'),
                            Item('scan_offset'),
                            label='Confocal scan',
                        ),

                        VGroup(
                            Item('CountTimeMonitor'),
                            Item('ReadoutDelayMonitor'),
                            Item('CrossTalkCorrectionMonitor'),
                            HGroup(
                                Item('MeasureCrossTalkMonitor',
                                     show_label=False,
                                     editor=ButtonEditor(label_value='MeasureCrossTalkLabelMonitor')
                                     ),
                                Item('MeasureCrossTalkVelocityMonitor'),
                                Item('MeasureCrossTalkResolutionMonitor'),
                            ),
                            label='Confocal Scan (Monitor position)',
                        ),
                    ),
                    VGroup(
                        Item('XYSize'),
                        Item('XYStep'),
                        Item('XYSettlingTime'),
                        Item('XYCountTime'),
                        Item('XOffset'),
                        Item('YOffset'),
                        Item('ZSize'),
                        Item('ZStep'),
                        Item('ZSettlingTime'),
                        Item('ZCountTime'),
                        Item('XYFitMethod'),
                        Item('ZFitMethod'),
                        label='Refocus (Tracker)', ),
                    VGroup(
                        Item('aom_voltage', visible_when='nidaq.aom_volt_ch!=None'),
                        label='General', ),
                    label='Settings'
                ), ),
            HGroup(Item(
                'state', style='custom', show_label=False,
                editor=EnumEditor(
                    values={
                        'idle': '1:idle',
                        'scan': '3:scan',
                        'refocus_crosshair': '4:refocus crosshair',
                        'refocus_trackpoint': '5:refocus trackpoint',
                        'xyz_monitor_scan': '6:xyz_monitor_scan',
                    },
                    cols=5
                ),
            ),
            ),
        ),
        title='Confocal',
        width=920,
        height=750,
        buttons=['OK'],
        resizable=True,
        x=0,
        y=0,
        id='ConfocalView')

    TraceView = View(
        VGroup(
            HGroup(
                Item('TraceLength', enabled_when='counter_state=="idle"'),
                Item('CountInterval', enabled_when='counter_state=="idle"'),
                Item('ReadoutInterval'),
            ),
            Item('counter_state', style='custom', show_label=False),
            Item('TracePlot', editor=ComponentEditor(), show_label=False, resizable=True),
        ),
        title='Counter',
        width=600,
        height=434,
        resizable=True,
        x=0,
        y=750,
        id='TraceView'
    )

    TrackerView = View(Tabbed(VGroup(  # Item('TrackImagePlot', editor=ComponentEditor(), show_label=False, resizable=True),
        # Item('TrackLinePlot', editor=ComponentEditor(), show_label=False, resizable=True),
        Item('TrackPlotContainer', editor=ComponentEditor(), show_label=False, resizable=True),
        HGroup(Item('periodic_refocus'),
               Item('TrackInterval'),
               Item('SetTrackPoint', show_label=False), ),
        HGroup(  # Item('TargetMode'),
            Item('AddTargetPoint', show_label=False),
            Item('RemoveTargetPoint', show_label=False),
            Item('CurrentTargetIndex', width=-40),
            Item('next_target_btn', show_label=False),
            Item('previous_target_btn', show_label=False)),
        HGroup(Item('CurrentTrackPoint', style='readonly'),
               Item('CurrentDrift', style='readonly'), ),
        label='Tracker'),
        Item('DriftPlot', editor=ComponentEditor(), show_label=False, resizable=True,
             label='Drift'), ),
        title='Tracker', width=280, height=550, x=915, y=0, buttons=['OK'], resizable=True,
        id='TrackerView')

    def __getstate__(self):
        """Returns current state of a selection of traits.
        Overwritten HasTraits.
        """
        state = SingletonHasTraits.__getstate__(self)

        for key in ['threads', 'state2method', 'count_thread', 'cross_talk_shift', 'scan_data_container']:
            if state.has_key(key):
                del state[key]

        return state
