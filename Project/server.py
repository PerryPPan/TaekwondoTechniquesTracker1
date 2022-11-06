import os
from flask import Flask, request, abort, jsonify
import database
import PoseProccessing



# Two unrelated videos have 91% error which is good means code works

#initialize flask model
app = Flask(__name__)

@app.route("/test")
def hello():
    return "Hello!"

#map specific url
app.config['UPLOAD_EXTENSIONS'] = ['.mpeg', 'mp4', 'mov']
@app.route('/video', methods=['GET', 'POST'])
def edit_video():
    # #receiving request + storing into variables
    video1 = request.files.getlist("video1")[1]
    video2 = request.files.getlist("video2")[1]
    #
    # #getting filename
    # video1_filename = uploaded_video1.filename
    # video2_filename = uploaded_video2.filename
    #
    # #_____altering filename ....
    # if video1_filename != '' and video2_filename != '':
    #     # _, video_file_ext = os.path.splitext(video1_filename)
    #     # _, image_file_ext = os.path.splitext(video2_filename)
    #     # if image_file_ext not in app.config['UPLOAD_EXTENSIONS'] or video_file_ext not in app.config[
    #     #     'UPLOAD_EXTENSIONS']:
    #     #     abort(400)
    #
    #     # Saving with new filename
    #     uploaded_video1.save(video1_filename)
    #     uploaded_video2.save(video2_filename)

    # You'll use the video1_path and video2_path to run the AI Engine
    # Call the function to run the AI Engine and returns the list of links to be access in the FrontEnd(Flutter APP)

    data = PoseProccessing.main(video1, video2)
    vid_results = database.send_images_example(video1, video2)
    print(data)
    print(vid_results)

    #remove/delete filepath
    # os.remove(video1_filename)
    # os.remove(video2_filename)
    # print("vid_results", vid_results)
    print("data", data)
    # print(vid_results, data)
    #python helper method to return JSON data
    # return jsonify(vid_results, data)

    return jsonify(vid_results ,data)

#run server locally
app.run(host = '0.0.0.0')