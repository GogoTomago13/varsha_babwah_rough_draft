import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
	
# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# create video output object 
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
	
# load the object detection network
# net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)






# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	faces = net.Process(img, opt.overlay)

	for face in faces:
		print(face)
		print(face.Keypoints)
		print('Links', face.Links)
		#varsha_left_idx = face.FindKeypoint(net.FindKeypointID('varsha_left'))

		#varsha_left = face.Keypoints[varsha_left_idx]

		#varsha_left_xcoord = varsha_left.x

		#varsha_left_ycoord = varsha_left.y

   
	# detect objects in the image (with overlay)
	#detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	#print("detected {:d} objects in image".format(len(detections)))

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
