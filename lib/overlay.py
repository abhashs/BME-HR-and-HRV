import datetime
import cv2


highest_bpm = 0
highest_bpms = []
col = (100, 255, 100)

def bpm_overlay(bpm, skips, frame_out2):
	if (skips == 30 and not (bpm == 50.3 or bpm == 50.4)):
		# find singular highest bpm by sample
		if (bpm > highest_bpm):
			highest_bpm = bpm
			timestamp_of_highest = datetime.datetime.now().strftime("%H:%M:%S")

		if (len(highest_bpms) < 5):
			highest_bpms.append((bpm, datetime.datetime.now().strftime("%H:%M:%S")))
			highest_bpms.sort(reverse=True)
		else:
			if (bpm > highest_bpms[-1][0]):
				highest_bpms.pop(len(highest_bpms) - 1)
				highest_bpms.append((bpm, datetime.datetime.now().strftime("%H:%M:%S")))
				highest_bpms.sort(key=lambda bpm_tuple: bpm_tuple[0], reverse=True)
		# print(self.highest_bpms)
		skips = 0
	skips += 1

	max_tracking_text = f"Heart Rate Peak: {str(round(highest_bpm, 1))} bpm at {timestamp_of_highest}"
	cv2.putText(frame_out2, max_tracking_text, (308, 415), cv2.FONT_HERSHEY_PLAIN, 1, col)
