# Reconstruction Sanity Report

## Aligned vs Random-Pair Control
- Obstacle IoU: aligned=0.9404, random_pair=0.3613, delta=0.5792
- Obstacle Dice: aligned=0.9628, random_pair=0.4749, delta=0.4879
- Geometry mIoU(3-class): aligned=0.9740, random_pair=0.7787, delta=0.1953

## Low-Quality Tail
- Cases with obstacle IoU < 0.50: 1/72 (1.39%)
- Cases with obstacle IoU < 0.70: 5/72 (6.94%)
- Shape accuracy on this holdout: 0.9861
- Corr(shape_confidence, obstacle_iou): 0.1803
- Low-confidence flag (<0.45): 1/72 flagged; 1/1 flagged cases have IoU<0.7

## Worst Cases (Top 10 by lowest obstacle IoU)
- case_id=airfoil_Re300_p07, shape_true=airfoil, shape_pred=circle, Re=300, iou_obstacle=0.1481, miou3=0.7039
- case_id=airfoil_Re200_p15, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.6667, miou3=0.8819
- case_id=airfoil_Re200_p25, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.6667, miou3=0.8888
- case_id=airfoil_Re100_p20, shape_true=airfoil, shape_pred=airfoil, Re=100, iou_obstacle=0.6667, miou3=0.8888
- case_id=airfoil_Re200_p14, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.6667, miou3=0.8888
- case_id=airfoil_Re300_p02, shape_true=airfoil, shape_pred=airfoil, Re=300, iou_obstacle=0.7500, miou3=0.9013
- case_id=airfoil_Re200_p28, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.7500, miou3=0.9166
- case_id=triangle_Re100_p27, shape_true=triangle, shape_pred=triangle, Re=100, iou_obstacle=0.7500, miou3=0.9082
- case_id=square_Re100_p11, shape_true=square, shape_pred=square, Re=100, iou_obstacle=0.8462, miou3=0.9364
- case_id=circle_Re100_p17, shape_true=circle, shape_pred=circle, Re=100, iou_obstacle=0.8621, miou3=0.9431