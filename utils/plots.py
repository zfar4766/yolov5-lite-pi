import cv2
import numpy as np


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    使用OpenCV画一个边界框
    Args:
        x: 边界框坐标 [x1, y1, x2, y2]
        img: 要画框的图像
        color: 框的颜色
        label: 标签文本
        line_thickness: 线条粗细
    """
    # 初始化线条粗细
    tl = line_thickness or max(round(sum(img.shape) / 2 * 0.003), 2)
    
    # 确保坐标是整数
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # 画矩形
    cv2.rectangle(img, c1, c2, color, thickness=max(tl - 1, 1), lineType=cv2.LINE_AA)
    
    # 如果有标签，添加文本
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        fs = max(tl - 1, 1) * 0.8  # 字体大小
        
        # 获取文本大小
        t_size = cv2.getTextSize(label, 0, fontScale=fs, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 3
        
        # 画标签背景
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        
        # 画标签文本
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, fs, [225, 255, 255], 
                    thickness=tf, lineType=cv2.LINE_AA)
