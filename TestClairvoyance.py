import MarkVideo


model = []
model.append(['weights\\weight-sion.npz', 'Sion'])
model.append(['weights\\weight-sejuani.npz', 'Sejuani'])
model.append(['weights\\weight-thresh.npz', 'Thresh'])
model.append(['weights\\weight-xayah.npz', 'Xayah'])
#model.append(['weights\\weight-jinx.npz', 'Jinx'])
#model.append(['weights\\weight-leesin.npz', 'Lee Sin'])
#model.append(['weights\\weight-ezreal.npz', 'Ezreal'])
#model.append(['weights\\weight-bard.npz', 'Bard'])

MarkVideo.MarkVideo('videos\\rawfeed.webm', 'videos\\test.avi', model, startFrame=0, frameSkip=6, numFrames=-1, windowSize=160, step=40, threshold=0.9)
#MarkVideo.GetOutputsFromVideo('videos\\rawfeed.webm', 'images\\result', model, startFrame=0, frameSkip=6, numFrames=210, windowSize=160, step=40, threshold=0.8)