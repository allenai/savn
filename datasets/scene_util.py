def make_scene_name(type, num):
    if type == "":
        return "FloorPlan" + str(num)
    elif num < 10:
        return "FloorPlan" + type + "0" + str(num)
    else:
        return "FloorPlan" + type + str(num)


def get_scenes(scene_str):
    scene_str_split = scene_str.split("+")
    if len(scene_str_split) == 1:
        return scene_str_split[0][1:-1].split(",")
    else:
        pre = scene_str_split[0][1:-1].split(",")
        for i in range(len(pre)):
            if pre[i] == "0":
                pre[i] = ""

        post = scene_str_split[1][1:-1].split("-")
        scene_names = [
            [make_scene_name(j, i) for i in range(int(post[0]), int(post[1]) + 1)]
            for j in pre
        ]
        # flatten list of lists to list
        out = [i + "_physics" for s in scene_names for i in s]
        new_out = []
        for k in out:
            if ("n3" in k or "n4" in k) and len(k) == 20:
                new_out.append(k[:12])
            else:
                new_out.append(k)
        return new_out
