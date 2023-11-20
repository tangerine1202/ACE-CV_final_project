import numpy as np


def read_nvm_file(file_name, return_rgb=False):
    # ref: FocusTune code
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    nb_cameras = int(lines[2])
    image2info = {}
    image2pose = {}
    image2name = {}
    for i in range(nb_cameras):
        cam_info = lines[3 + i]
        img_name, info = cam_info.split("\t")
        focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = map(float, info.split(" "))
        image2name[i] = img_name
        image2info[i] = [focal, radial]
        image2pose[i] = [qw, qx, qy, qz, tx, ty, tz]
    nb_points = int(lines[4 + nb_cameras])
    image2points = {}
    image2uvs = {}
    xyz_arr = np.zeros((nb_points, 3), np.float64)
    rgb_arr = np.zeros((nb_points, 3), np.uint8)
    for j in range(nb_points):
        point_info = lines[5 + nb_cameras + j].split(" ")
        x, y, z, r, g, b, nb_features = map(float, point_info[:7])
        xyz_arr[j] = [x, y, z]
        rgb_arr[j] = [r, g, b]
        features_info = point_info[7:]
        nb_features = int(nb_features)
        for k in range(nb_features):
            image_id, feature_id, u, v = features_info[k * 4: (k + 1) * 4]
            image_id, feature_id = map(int, [image_id, feature_id])
            u, v = map(float, [u, v])
            image2points.setdefault(image_id, []).append(j)
            image2uvs.setdefault(image_id, []).append([u, v])

    if return_rgb:
        return xyz_arr, image2points, image2name, rgb_arr
    else:
        return xyz_arr, image2points, image2name
