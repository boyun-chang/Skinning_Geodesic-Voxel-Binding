import maya.cmds as cmds
import maya.api.OpenMaya as om

def get_world_transforms():
    # 모든 joint 찾기
    joints = cmds.ls(type='joint')
    world_transforms = {}

    for joint in joints:
        # Joint의 월드 스페이스 트랜스폼 매트릭스 가져오기
        world_matrix = cmds.xform(joint, q=True, ws=True, m=True)
        # 월드 매트릭스를 사용하여 위치(Position) 계산
        mMatrix = om.MMatrix(world_matrix)
        mTransformMatrix = om.MTransformationMatrix(mMatrix)
        position = mTransformMatrix.translation(om.MSpace.kWorld)

        # 결과 저장
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