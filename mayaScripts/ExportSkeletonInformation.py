import maya.cmds as cmds
import maya.api.OpenMaya as om

def get_world_transforms():
    # Find all joints
    joints = cmds.ls(type='joint')
    world_transforms = {}

    for joint in joints:
        # Get the world space transform matrix of the joint
        world_matrix = cmds.xform(joint, q=True, ws=True, m=True)
        # Calculating position using world matrix
        mMatrix = om.MMatrix(world_matrix)
        mTransformMatrix = om.MTransformationMatrix(mMatrix)
        position = mTransformMatrix.translation(om.MSpace.kWorld)

        # Save results
        world_transforms[joint] = position

    return world_transforms

def save_transforms_to_txt(filename, transforms):
    with open(filename, 'w') as f:
        for joint, position in transforms.items():
            f.write(f"{joint} {position}\n")
    print(f"Transforms saved to {filename}")

# 실행 예제
transforms = get_world_transforms()
save_transforms_to_txt("D:/maya/projects/cgxr/scripts/skeleton_transforms.txt", transforms)
