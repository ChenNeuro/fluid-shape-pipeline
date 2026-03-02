# Reconstruction Sanity Report

## Aligned vs Random-Pair Control
- Obstacle IoU: aligned=0.9463, random_pair=0.2982, delta=0.6481
- Obstacle Dice: aligned=0.9685, random_pair=0.4244, delta=0.5441
- Geometry mIoU(3-class): aligned=0.9763, random_pair=0.7562, delta=0.2201

## Low-Quality Tail
- Cases with obstacle IoU < 0.50: 2/90 (2.22%)
- Cases with obstacle IoU < 0.70: 7/90 (7.78%)
- Shape accuracy on this holdout: 1.0000
- Corr(shape_confidence, obstacle_iou): -0.0010
- Low-confidence flag (<0.45): 0/90 flagged; 0/1 flagged cases have IoU<0.7

## Worst Cases (Top 10 by lowest obstacle IoU)
- case_id=airfoil_Re300_p02, shape_true=airfoil, shape_pred=airfoil, Re=300, iou_obstacle=0.5000, miou3=0.8152
- case_id=airfoil_Re200_p15, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.5000, miou3=0.8332
- case_id=airfoil_Re100_p20, shape_true=airfoil, shape_pred=airfoil, Re=100, iou_obstacle=0.6667, miou3=0.8750
- case_id=airfoil_Re300_p06, shape_true=airfoil, shape_pred=airfoil, Re=300, iou_obstacle=0.6667, miou3=0.8807
- case_id=airfoil_Re100_p07, shape_true=airfoil, shape_pred=airfoil, Re=100, iou_obstacle=0.6667, miou3=0.8861
- case_id=airfoil_Re100_p19, shape_true=airfoil, shape_pred=airfoil, Re=100, iou_obstacle=0.6667, miou3=0.8861
- case_id=airfoil_Re200_p28, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.6667, miou3=0.8874
- case_id=airfoil_Re200_p09, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.7500, miou3=0.9096
- case_id=airfoil_Re200_p27, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.7500, miou3=0.9125
- case_id=airfoil_Re200_p14, shape_true=airfoil, shape_pred=airfoil, Re=200, iou_obstacle=0.7500, miou3=0.9152