import math
import numpy as np
import time
import cv2
import pylab
import os
import sys
import datetime
import csv
import time
import functools
import matplotlib.pyplot as plt

from lib.overlay import bpm_overlay

from regex import W
import scipy
import scipy.signal
import scipy.stats

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.frame_in = np.zeros((10, 10))
        self.frame_in2 = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.frame_out2 = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        self.filename = ""
        self.filebool = False
        self.count = 0
        self.start_time = time.time()
        self.face_in_frame = False
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.highest_bpm = 0
        self.highest_bpms = []
        self.timestamp_of_highest = ""
        self.skips = 0
        self.hrv_skips = 0
        self.hrv_samples = []
        self.hrv_time = []
        self.bpms = []
        self.time = None
        self.hrv = 60

        self.idx = 1
        self.find_faces = True

    def get_face_in_frame(self):
        return face_in_frame

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def train(self):
        self.trained = not self.trained
        return self.trained

    def partition(self, list, indices):
        partitions = []
        for index in indices:
            partition = list[:index]
            partitions.append(partition)
            list = list[index:]
        return partitions

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.frame_out2 = self.frame_in2
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                    cam),
                (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                       (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                       (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            self.data_buffer, self.times, self.trained = [], [], False
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))
            if len(detected) > 0:
                self.face_in_frame = True
                detected.sort(key=lambda a: a[-1] * a[-2])
                
                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
            else:
                self.face_in_frame = False
                    
            forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            self.draw_rect(self.face_rect, col=(255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                       (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            cv2.putText(self.frame_out, "Forehead",
                       (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        cv2.putText(
            self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                cam),
            (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
        cv2.putText(
            self.frame_out, "Press 'S' to restart",
                   (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                   (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        
        ############
        #ADDED CODE#
        ############
        detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            self.face_in_frame = True
            detected.sort(key=lambda a: a[-1] * a[-2])

            if self.shift(detected[-1]) > 10:
                self.face_rect = detected[-1]
        else:
            self.face_in_frame = False
                
        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(self.face_rect, col=(255, 0, 0))
        x, y, w, h = self.face_rect
        cv2.putText(self.frame_out, "",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        self.draw_rect(forehead1)
        x, y, w, h = forehead1
        cv2.putText(self.frame_out, "",
                       (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        ###################
        #END OF ADDED CODE#
        ###################

        ##################
        #Virtual Device
        ################
        origin = (100, 400) # (0, 0)

        cv2.ellipse(self.frame_out2, origin, (90, 90), 0, -180, 0, col, 2)       # Dial frame
        cv2.line(self.frame_out2, (10, 400), (190, 400), col, 2)                 # Dial lower frame
        cv2.ellipse(self.frame_out2, origin, (10, 10), 0, -180, 0, col, -1)      # Dial lower adornment

        # percentage = self.bpm/180;
        # angle = math.sin
        hrv_value = self.hrv
        if (self.hrv < 20):
            hrv_value = 20
        elif (self.hrv > 200):
            hrv_value = 200

        hrv_value = 200 - hrv_value
        if (hrv_value < 20):
            hrv_value = 20

        hrv_degrees = hrv_value - 20
        end_pointX = (90 * math.cos(math.radians(180 - hrv_degrees))) + 100
        end_pointY =  400 - (90 * math.sin(math.radians(hrv_degrees)))
        # print((end_pointX, end_pointY))

        cv2.line(self.frame_out2, origin, (int(end_pointX), int(end_pointY)), col, 2)
        
        cv2.putText(self.frame_out2, "0%", (10, 415), cv2.FONT_HERSHEY_PLAIN, 1, col)
        cv2.putText(self.frame_out2, "100%", (150, 415), cv2.FONT_HERSHEY_PLAIN, 1, col)
        cv2.putText(self.frame_out2, "Stress Estimate", (35, 300), cv2.FONT_HERSHEY_PLAIN, 1, col)

        # bpm_overlay(self.bpm, self.skips, self.frame_out2)

        if (self.skips == 30 and not (self.bpm == 50.3 or self.bpm == 50.4)):
            # find singular highest bpm by sample
            if (self.bpm > self.highest_bpm):
                self.highest_bpm = self.bpm
                self.timestamp_of_highest = datetime.datetime.now().strftime("%H:%M:%S")

            if (len(self.highest_bpms) < 5):
                self.highest_bpms.append((self.bpm, datetime.datetime.now().strftime("%H:%M:%S")))
                self.highest_bpms.sort(reverse=True)
            else:
                if (self.bpm > self.highest_bpms[-1][0]):
                    self.highest_bpms.pop(len(self.highest_bpms) - 1)
                    self.highest_bpms.append((self.bpm, datetime.datetime.now().strftime("%H:%M:%S")))
                    self.highest_bpms.sort(key=lambda bpm_tuple: bpm_tuple[0], reverse=True)
            # print(self.highest_bpms)
            self.skips = 0
        self.skips += 1

        max_tracking_text = f"Heart Rate Peak: {str(round(self.highest_bpm, 1))} bpm at {self.timestamp_of_highest}"
        cv2.putText(self.frame_out2, max_tracking_text,
                    (308, 415), cv2.FONT_HERSHEY_PLAIN, 1, col)


        #forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        #self.draw_rect(forehead1)

        vals = self.get_subface_means(forehead1)

        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        
        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))
            # print(idx)

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]



            # new_time = round(time.time() * 1000)
            # self.bpms.append(self.bpm)
            # self.hrv_samples.append(t)
            # self.hrv_time.append(new_time - self.time)
            # self.time = new_time

            # rr_intervals = list(map(lambda bpm: 6000/bpm, self.bpms))

            # count = 1
            # prev_rr = rr_intervals[1]
            # prev_calc = math.pow(rr_intervals[0] - rr_intervals[1], 2)
            # hrv = prev_calc
            # for rr in rr_intervals[2:]:
            #     hrv += math.pow(prev_rr - rr, 2)
            #     count += 1

            # self.hrv = math.sqrt(hrv/count)
            # print(self.hrv)

            # HRV sampling and calculations
            if self.time == None:
                self.time = round(time.time() * 1000)

            if (self.hrv_skips < 150):
                self.hrv_skips += 1
                new_time = round(time.time() * 1000)
                self.bpms.append(self.bpm)
                self.hrv_samples.append(t)
                self.hrv_time.append(new_time - self.time)
                self.time = new_time
            else:
                peak_indices, _ = scipy.signal.find_peaks(self.hrv_samples, height=.6)
                peaks_at_index = []
                for peak in peak_indices:
                    peaks_at_index.append(self.hrv_samples[peak])
                
                # print(peaks_at_index)


                pulses = self.partition(self.hrv_time, peak_indices);
                pulses = list(filter(lambda pulse: pulse != [], pulses))

                rr_intervals = []
                for pulse in pulses:
                    # print(sum(pulse), len(pulse) * 33)
                    rr_intervals.append(len(pulse) * 33) # num of frames * duration of frame in ms
                    # rr_intervals.append(sum(pulse))


                # filter out noise values 
                rr_intervals = list(filter(lambda rr: rr > 600 and rr < 1200, rr_intervals))
                if (len(rr_intervals) > 1):
                    zscores = scipy.stats.zscore(rr_intervals)
                    median = np.median(rr_intervals)
                    # print(zscores)
                    for index, rr in enumerate(rr_intervals):
                        zscore = zscores[index]
                        if np.abs(zscore) > 1.44:   #85 confidence
                            rr_intervals[index] = median
                    
                    # print(rr_intervals)
                    rmssd_hrv = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
                    self.hrv = rmssd_hrv or 60
                print(self.hrv)
                
                # rr_intervals2[np.abs(scipy.stats.zscore(rr_intervals)) > 2] = np.median(rr_intervals)
                # print(rmssd)

                # if (len(rr_intervals) >= 3):
                #     count = 1
                #     prev_rr = rr_intervals[1]
                #     prev_calc = math.pow(rr_intervals[0] - rr_intervals[1], 2)
                #     hrv = prev_calc
                #     for rr in rr_intervals[2:]:
                #         hrv += math.pow(prev_rr - rr, 2)
                #         count += 1
                #     self.hrv = math.sqrt(hrv/count)
                #     print(self.hrv)

                # fig = plt.figure()
                # ax = fig.subplots()
                # ax.plot(list(range(1, 151)), self.hrv_samples)
                # ax.scatter(peak_indices, peaks_at_index, color ="r")
                # plt.show()

                # BPM FORMULA ALTERNATIVE
                # rr_intervals = list(map(lambda bpm: 6000/bpm, self.bpms))

                # count = 1
                # prev_rr = rr_intervals[1]
                # prev_calc = math.pow(rr_intervals[0] - rr_intervals[1], 2)
                # hrv = prev_calc
                # for rr in rr_intervals[2:]:
                #     hrv += math.pow(prev_rr - rr, 2)
                #     count += 1

                # self.hrv = math.sqrt(hrv/count)
                # print(self.hrv)

                self.hrv_skips = 0
                self.hrv_samples = []
                self.hrv_time = []
                self.time = round(time.time() * 1000)

            # if (np.amax(g) > 251):
                # print("beat")

            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps
            
            #initialize csv file
            if(self.filebool == False):
                self.filebool = True
                f = str(datetime.datetime.now())
                self.filename = "test" + ".csv"
                with open(self.filename, 'w') as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow(['Raw', 'BPM', 'In Frame', 'Time (s)'])
            
            #write to csv file
            with open(self.filename, 'a') as f:
                writer = csv.writer(f)
                row = [str(self.samples[-1]), str(self.bpm), str(self.face_in_frame), str(time.time() - self.start_time)]
                writer.writerow(row)
                #self.count = self.count + 1

            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
            tsize = 1
            cv2.putText(self.frame_out, text,
                       (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)

            cv2.putText(self.frame_out2, "Heart Rate Estimate: %0.1f bpm" % (self.bpm) if self.bpm > 55 else "Heart Rate Estimate: Inconclusive",
                       (308, 385), cv2.FONT_HERSHEY_PLAIN, tsize, col)

                
            cv2.putText(self.frame_out2, "HRV Estimate: %0.1f ms" % (self.hrv) if (self.hrv > 20 and self.hrv < 220) else "HRV Estimate: Calculating...", (308, 400), cv2.FONT_HERSHEY_PLAIN, tsize, col)
