from decord import VideoReader, cpu
import pandas as pd
import cv2
import numpy as np


def process_frame(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gauss = cv2.GaussianBlur(gray, (21, 21), 3)
	threshold = cv2.threshold(gauss, 108, 255, cv2.THRESH_BINARY)[1]
	return threshold


def main():
	vl = VideoReader('videos/bee.mp4', ctx=cpu(0))

	df = pd.read_pickle('pickles/WaggleDetections-bee_5M-Cleaned.pkl')
	uniques = df['cluster'].unique()

	for u in uniques:
		frames = df.loc[df['cluster'] == u, 'frame']
		batch = vl.get_batch(frames).asnumpy()

		# print(f'Cluster: {u} \nFrames:\n{frames}')
		print(f'Frame Stack: {batch.shape}')

		processed_batch = [process_frame(batch[frame]) for frame in range(batch.shape[0])]


		# processed_batch = np.apply_over_axes(process_frame, batch, [1,2])

		print(f'Processed Batch: {processed_batch[0]}')

		if cv2.waitKey(0) == 27:
			break

	cv2.destroyAllWindows()

	# allFrames = df['frame']
	# duplicate_mask = df.frame.duplicated()
	# print(df[duplicate_mask])

	# print(uniques)

    # split_frames()

if __name__ == '__main__':
    main()
