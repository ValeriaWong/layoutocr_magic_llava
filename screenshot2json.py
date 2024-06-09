import argparse
import json
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine
import requests
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

class ImageTools:
    @staticmethod
    def calculate_iou(box1, box2):
        """计算两个边界框的交并比(IoU)。"""
        x1, y1 = max(box1['min_x'], box2['min_x']), max(
            box1['min_y'], box2['min_y'])
        x2, y2 = min(box1['max_x'], box2['max_x']), min(
            box1['max_y'], box2['max_y'])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1['max_x'] - box1['min_x']) * \
            (box1['max_y'] - box1['min_y'])
        box2_area = (box2['max_x'] - box2['min_x']) * \
            (box2['max_y'] - box2['min_y'])
        return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) else 0

    @staticmethod
    def merge_boxes(box1, box2):
        """合并两个边界框，并返回一个包含两者的最大边界框。"""
        return {
            'min_x': min(box1['min_x'], box2['min_x']),
            'max_x': max(box1['max_x'], box2['max_x']),
            'min_y': min(box1['min_y'], box2['min_y']),
            'max_y': max(box1['max_y'], box2['max_y'])
        }

    @staticmethod
    def calculate_histogram(image, mask=None):
        """计算图像的颜色直方图。"""
        histogram = cv2.calcHist([image], [0, 1, 2], mask, [
                                 8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)
        return histogram.flatten()

    @staticmethod
    def compare_histograms(hist1, hist2):
        """比较两个直方图的相似度，使用余弦相似度。"""
        return cosine(hist1, hist2)

    @staticmethod
    def crop_images_by_boxes(image, boxes):
        """根据边界框从原始图像裁剪头像图像，并返回图像、边界框及其索引代号。"""
        cropped_data = []
        for index, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = int(box['min_x']), int(
                box['min_y']), int(box['max_x']), int(box['max_y'])
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_data.append({
                'image': cropped_image,
                'box': box,
                'id': f"avatar_{index}"
            })
        return cropped_data

    @staticmethod
    def calculate_horizontal_distance(box1, box2):
        """计算两个边界框之间的水平距离。"""
        distance = min(abs(box1['max_x'] - box2['min_x']),
                       abs(box1['min_x'] - box2['max_x']))
        return distance

    @staticmethod
    def calculate_vertical_distance(box1, box2):
        """计算两个边界框之间的垂直距离。"""
        vertical_distance = min(
            abs(box1['min_y'] - box2['max_y']), abs(box1['max_y'] - box2['min_y']))
        return vertical_distance
    
    @staticmethod
    def parse_timestamp(timestamp_text):
        """
        尝试根据不同的格式解析时间戳文本。
        """
        # 定义可能的时间戳格式
        formats = [
            "%Y年%m月%d日 %H:%M",  # 完整的年月日时间
            "%Y年%m月%d日",         # 只有年月日
            "%m月%d日 %H:%M",      # 没有年份的月日和时间
            "%m月%d日",            # 只有月和日
            "%H:%M"                # 只有时间
        ]

        # 当前日期作为默认年月日
        now = datetime.now()
        default_year = now.year
        default_month = now.month
        default_day = now.day

        for fmt in formats:
            try:
                # 尝试用每一个格式进行解析
                parsed_date = datetime.strptime(timestamp_text, fmt)
                # 如果年份没有提供，使用当前年份
                if '%Y' not in fmt:
                    parsed_date = parsed_date.replace(year=default_year)
                # 如果月份和日期没有提供，使用当前月份和日期
                if '%m' not in fmt or '%d' not in fmt:
                    parsed_date = parsed_date.replace(
                        month=default_month, day=default_day)
                return parsed_date
            except ValueError:
                continue  # 当前格式不匹配，尝试下一个格式

        # 如果所有格式都无法解析，返回None
        return None

    def upload_image( url, image_path):
        """
            上传图片到指定的URL。

            :param url: API端点的URL。
            :param image_path: 要上传的图片文件的路径。
            :return: API的响应。
            """
        # 打开文件
        with open(image_path, 'rb') as image_file:
            # 文件使用files参数传递
            files = {'file': (image_path, image_file, 'image/jpeg')}

            # 发送POST请求
            response = requests.post(url, files=files)

            return response

    def is_unique_avatar(current_hist, all_histograms, threshold=0.2):
        """
        判断当前头像是否为独一无二的。
        """
        for hist in all_histograms:
            if ImageTools.compare_histograms(current_hist, hist) < threshold:
                return False
        return True

    def count_unique_avatars_by_content(images):
        """
        根据内容判断独特头像的数量。
        """
        unique_histograms = []
        for image in images:
            current_hist = ImageTools.calculate_histogram(image)
            if ImageTools.is_unique_avatar(current_hist, unique_histograms):
                unique_histograms.append(current_hist)
        return len(unique_histograms)

    def crop_and_calculate_histograms(image, avatars):
        """ 裁剪图像并计算直方图 """
        avatars_with_histograms = []
        for avatar in avatars:
            cropped_avatar = ImageTools.crop_images_by_boxes(
                image, [avatar['bbox']])
            avatar_hist = ImageTools.calculate_histogram(
                cropped_avatar[0]['image'])
            avatars_with_histograms.append({
                'avatar': avatar,
                'histogram': avatar_hist
            })
        return avatars_with_histograms

    def identify_avatar(current_hist, all_histograms):
        """ 识别头像是否独一无二并赋予ID """
        threshold = 0.2  # 相似度阈值
        for index, hist in enumerate(all_histograms):
            if ImageTools.compare_histograms(current_hist, hist) < threshold:
                return f"avatar_{index}"
        return f"unique_{len(all_histograms)}"  # 为独一无二的头像赋予唯一ID

    def find_leftmost_header(extracted_texts):
        """
        查找页面中最靠左的头部文本作为昵称。
        """
        leftmost_text = None
        min_x = float('inf')
        for text_info in extracted_texts:
            if text_info['assigned_label'] == 'header' and text_info['bounding_box']['min_x'] < min_x:
                min_x = text_info['bounding_box']['min_x']
                leftmost_text = text_info['text']
        return leftmost_text


class ChatOrganizer:
    def __init__(self, image_path, config_file, checkpoint_file):
        self.image_path = image_path
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_detector(
            config_file, checkpoint_file, device='cuda:0')
        self.extracted_texts = []
        self.detected_objects = []
        self.dialog_type = None
        self.dialog_name = None

    def extract_text(self, image):
        """使用OCR技术提取图像中的文字及其边界框"""
        # API的URL
        api_url = os.getenv(
            'API_URL')
        # 调用函数
        response = ImageTools.upload_image(api_url, self.image_path)
        # 打印响应内容
        # print('Status Code:', response.status_code)
        # print('Response Body:', response.text)

        # 将JSON字符串解析为字典
        ocr_data = json.loads(response.text)
        # 准备收集每个句子的文本和边界框
        sentences = []

        # 遍历每个单词，构建句子和计算边界框
        for page in ocr_data['responses'][0]['fullTextAnnotation']['pages']:
            for block in page['blocks']:
                if block['blockType'] == 'TEXT':
                    for paragraph in block['paragraphs']:
                        sentence_text = []
                        sentence_bbox = {
                            'min_x': float('inf'), 'max_x': 0, 'min_y': float('inf'), 'max_y': 0
                        }

                        for word in paragraph['words']:
                            word_text = ''.join([symbol['text']
                                                for symbol in word['symbols']])
                            sentence_text.append(word_text)

                            vertices = word['boundingBox']['vertices']
                            for vertex in vertices:
                                sentence_bbox['min_x'] = min(
                                    sentence_bbox['min_x'], vertex.get('x', sentence_bbox['min_x']))
                                sentence_bbox['max_x'] = max(
                                    sentence_bbox['max_x'], vertex.get('x', sentence_bbox['max_x']))
                                sentence_bbox['min_y'] = min(
                                    sentence_bbox['min_y'], vertex.get('y', sentence_bbox['min_y']))
                                sentence_bbox['max_y'] = max(
                                    sentence_bbox['max_y'], vertex.get('y', sentence_bbox['max_y']))

                        # 合并单词形成句子，并将句子及其边界框存储
                        sentences.append({
                            'text': ''.join(sentence_text),
                            'bounding_box': sentence_bbox
                        })

        # 打印每个句子及其边界框
        for sentence in sentences:
            print(f"句子: {sentence['text']}")
            print(f"边界框: {sentence['bounding_box']}")
        return sentences

    def detect_objects(self, img, config_file, checkpoint_file, COCO_CLASSES):
        # 通过配置和权重初始化模型
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        result = inference_detector(model, img)
        pred_instances = result.pred_instances
        result_list = []
        # 提取边界框、标签和得分
        bboxes = pred_instances.bboxes.cpu().numpy()  # 将tensor转移到CPU并转为NumPy数组
        labels = pred_instances.labels.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        # 检查边界框数组是否非空
        if len(bboxes) > 0:
            # 遍历所有边界框和相应的标签与得分
            for bbox, label, score in zip(bboxes, labels, scores):
                if score > 0.5:
                    # TODO：clarify
                    min_x, min_y, max_x, max_y = bbox

                    # print(f"类别 {label}, {COCO_CLASSES[label]}:")
                    # print(
                    #     f"  边界框 [x_min: {min_x}, y_min: {min_y}, x_max: {max_x}, y_max: {max_y}], 得分：{score:.4f}")
                    result_list.append(
                        {"bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}, "label": COCO_CLASSES[label], "score": score})
            return result_list


    def assign_labels_to_texts(self,detected_objects, extracted_texts):
        """为每个OCR文本框分配最合适的属性，并检测可能的合并。"""
        for text_info in extracted_texts:
            text_box = text_info['bounding_box']
            highest_iou = 0
            assigned_label = None

            for obj in detected_objects:
                obj_label = obj['label']
                obj_box = obj['bbox']
                obj_score = obj['score']
                if obj_score > 0.5:
                    iou = ImageTools.calculate_iou(obj_box, text_box)
                    if iou > highest_iou:
                        highest_iou = iou
                        assigned_label = obj_label

            text_info['assigned_label'] = assigned_label if assigned_label else "unknown"

    def merge_texts_by_detection_boxes(self, extracted_texts, detected_objects):
        """使用检测到的message边界框来合并文本框。"""
        # 创建每个文本的最佳匹配message边界框
        for text_info in extracted_texts:
            best_match = None
            best_iou = 0
            for obj in detected_objects:
                if obj['label'] == 'message':
                    iou = ImageTools.calculate_iou(
                        text_info['bounding_box'], obj['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj['bbox']

            # 将最佳匹配边界框分配给文本
            text_info['message_bbox'] = best_match if best_match else text_info['bounding_box']

        # 按 y_min 排序文本以保持顺序
        extracted_texts.sort(key=lambda x: x['message_bbox']['min_y'])

        # 合并在同一message边界框内的文本框
        merged_texts = []
        current_text = None
        for text_info in extracted_texts:
            if not current_text:
                current_text = text_info
            else:
                # 判断当前文本是否应合并到前一个文本
                if current_text['message_bbox'] == text_info['message_bbox'] or \
                        ImageTools.calculate_iou(current_text['message_bbox'], text_info['message_bbox']) > 0:
                    current_text['text'] += "" + text_info['text']
                    # 更新当前文本的边界框为两者的联合
                    current_text['bounding_box'] = ImageTools.merge_boxes(
                        current_text['bounding_box'], text_info['bounding_box'])
                else:
                    merged_texts.append(current_text)
                    current_text = text_info

        if current_text:
            merged_texts.append(current_text)

        return merged_texts

    def determine_chat_type(self, extracted_texts, images, avator_boxes):
        """
        确定聊天类型：群聊或单聊，并尝试返回聊天名称。
        """
        dialog_name = "未检测到header属性"
        group_chat_detected = False

        # 检查是否有昵称直接位于消息之上
        for text_info in extracted_texts:
            if text_info['assigned_label'] == 'nickname':
                nickname_box = text_info['bounding_box']
                for message_info in extracted_texts:
                    if message_info['assigned_label'] == 'message':
                        message_box = message_info['bounding_box']
                        # 检查昵称是否在消息直接上方
                        if (nickname_box['max_y'] <= message_box['min_y'] and
                                abs(nickname_box['max_y'] - message_box['min_y']) < 10):
                            group_chat_detected = True
                            break
                if group_chat_detected:
                    break

        # 如果没有检测到群聊，检查头像多样性
        if not group_chat_detected:
            # 裁剪头像图像
            cropped_images = ImageTools.crop_images_by_boxes(
                images, avator_boxes)
            # 计算头像的多样性
            if ImageTools.count_unique_avatars_by_content([img['image'] for img in cropped_images]) > 2:
                group_chat_detected = True

        # 使用头像信息确定聊天类型
        if group_chat_detected:
            dialog_type = "群聊"
        else:
            dialog_type = "单聊"

        # 使用 header 中的信息确定聊天名称
        for text_info in extracted_texts:
            # print(text_info['assigned_label'], text_info['text'])
            if text_info['assigned_label'] == 'header' or text_info['assigned_label'] == 'unknown':
                if '(' in text_info['text'] or dialog_type == "单聊":
                    dialog_name = text_info['text']
                    break
        print("")
        return dialog_type, dialog_name


    def assign_nicknames_to_avatars(self,extracted_texts, avatars, dialog_type, image):
        """
        为每个头像分配昵称，考虑头像内容的相似性。
        - 群聊：头像的昵称是在垂直方向上与之对齐且水平距离最近的昵称。
        - 单聊：头像的昵称从左到右是对方的名字和截图者。
        """
        avatars_with_histograms = ImageTools.crop_and_calculate_histograms(
            image, avatars)
        histograms = [avatar['histogram'] for avatar in avatars_with_histograms]

        # 为头像分配ID和检查唯一性
        for avatar_info in avatars_with_histograms:
            avatar_info['avatar']['id'] = ImageTools.identify_avatar(
                avatar_info['histogram'], histograms)

        # 群聊中为头像分配昵称
        if dialog_type == "群聊":
            for avatar_info in avatars_with_histograms:
                nickname = self.match_nickname_by_proximity(
                    avatar_info['avatar'], extracted_texts, image.shape[1])
                avatar_info['avatar'][
                    'nickname'] = nickname if nickname else f"Unique_{avatar_info['avatar']['id']}"

        # 单聊情况处理
        else:
            if len(avatars_with_histograms) > 1:
                avatars_with_histograms.sort(
                    key=lambda x: x['avatar']['bbox']['min_x'])
                left_avatar = avatars_with_histograms[0]['avatar']
                right_avatar = avatars_with_histograms[-1]['avatar']
                left_avatar['nickname'] = ImageTools.find_leftmost_header(
                    extracted_texts)
                right_avatar['nickname'] = "我"

        return avatars_with_histograms

    def assign_nickname_to_avatar(self, avatar, extracted_texts):
        """
        为头像分配最近的昵称，基于垂直对齐和水平距离。
        """
        min_distance = float('inf')
        closest_nickname = None
        avatar_y_min = avatar['bbox']['min_y']
        for text_info in extracted_texts:
            if text_info['assigned_label'] == 'nickname':
                text_y_min = text_info['bounding_box']['min_y']
                # 检查垂直对齐
                if abs(avatar_y_min - text_y_min) < 10:  # 允许的对齐误差
                    distance = ImageTools.calculate_horizontal_distance(
                        avatar['bbox'], text_info['bounding_box'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_nickname = text_info['text']
        avatar['nickname'] = closest_nickname if closest_nickname else "未知"


    def match_nickname_by_proximity(self,avatar, extracted_texts, image_width):
        """
        通过水平距离和头像的位置来匹配昵称，同时考虑垂直对齐。
        """
        min_distance = float('inf')
        chosen_nickname = None
        avatar_center = (avatar['bbox']['min_x'] + avatar['bbox']['max_x']) / 2
        avatar_y_min = avatar['bbox']['min_y']
        for text in extracted_texts:
            if text['assigned_label'] == 'nickname':
                text_center = (text['bounding_box']['min_x'] +
                            text['bounding_box']['max_x']) / 2
                text_y_min = text['bounding_box']['min_y']
                if abs(avatar_y_min - text_y_min) < 10:  # 垂直对齐的容差
                    distance = abs(avatar_center - text_center)
                    if distance < min_distance:
                        min_distance = distance
                        chosen_nickname = text['text']

        # 考虑页面右侧的头像通常是截图者
        if avatar_center > image_width * 0.8:  # 假设页面右侧20%是截图者
            return "我"
        return chosen_nickname if chosen_nickname else self.find_matching_nickname_for_avatar(avatar, extracted_texts)

    def find_matching_nickname_for_avatar(self, avatar, extracted_texts):
        """
        如果没有直接对齐的昵称，尝试查找具有相同ID的其他头像的昵称。
        """
        avatar_id = avatar['id']
        for other_avatar in extracted_texts:
            if other_avatar.get('id') == avatar_id and 'nickname' in other_avatar:
                return other_avatar['nickname']
        return avatar_id



    def assign_speaker_to_messages(self, extracted_texts, avatars, dialog_type, dialog_name, image_width):
        """
        为每个消息分配说话人。
        - 群聊：优先使用左上方且左对齐的昵称，如果没有昵称，则根据头像位置分配。
        - 单聊：消息通常是由对话中的另一方发送，头像在右侧则可能是截图者。
        """
        for text_info in extracted_texts:
            if text_info['assigned_label'] == 'message':
                message_box = text_info['bounding_box']
                closest_avatar = None
                closest_nickname = None
                min_distance = float('inf')
                is_right_side = False
                # 先检查有头像的情况
                min_vertical_distance = float('inf')  # 重置最小垂直距离
                for avatar_info in avatars:
                    avatar_box = avatar_info['avatar']['bbox']
                    vertical_distance = abs(
                        avatar_box['min_y'] - message_box['min_y'])
                    if vertical_distance < min_vertical_distance:
                        min_vertical_distance = vertical_distance
                        closest_avatar = avatar_info
                        is_right_side = closest_avatar['avatar']['bbox']['min_x'] > image_width * 0.8
                    if closest_avatar and not is_right_side:
                        text_info['speaker'] = closest_avatar['avatar']['nickname']
                
                # 群聊情况下，检查是否有昵称紧邻消息(优先级大于上面使用头像位置的，会覆盖掉)
                # 群聊情况下，检查是否有昵称位于消息左上方且左对齐
                if dialog_type == "群聊" and not is_right_side:
                    for nickname_info in extracted_texts:
                        if nickname_info['assigned_label'] == 'nickname':
                            # 计算左对齐的距离（基于 min_x）
                            horizontal_alignment = abs(
                                nickname_info['bounding_box']['min_x'] - message_box['min_x'])
                            # 检查昵称是否在消息的上方
                            vertical_alignment = message_box['min_y'] - \
                                nickname_info['bounding_box']['max_y']

                            if horizontal_alignment < 10 and vertical_alignment > 0:
                                if vertical_alignment < min_distance:
                                    min_distance = vertical_alignment
                                    closest_nickname = nickname_info['text']
                    if closest_nickname and not is_right_side:
                        text_info['speaker'] = closest_nickname
                elif dialog_type == "单聊" and not is_right_side:
                    text_info['speaker'] = dialog_name
                if is_right_side:
                    text_info['speaker'] = "我"  # 默认为截图者





    def assign_timestamps_to_messages(self,extracted_texts):
        """
        为每条消息根据其相对于时间戳的位置分配时间。
        """
        timestamps = [
            text for text in extracted_texts if text['assigned_label'] == 'timestamp']
        messages = [
            text for text in extracted_texts if text['assigned_label'] == 'message']

        # 排序时间戳和消息按其在页面上的位置
        timestamps.sort(key=lambda x: x['bounding_box']['min_y'])
        messages.sort(key=lambda x: x['bounding_box']['min_y'])

        # 为消息分配时间戳
        if timestamps:
            last_assigned_timestamp = None
            current_timestamp_index = 0

            for message in messages:
                # 查找消息所属的时间段
                while current_timestamp_index < len(timestamps) - 1 and \
                        message['bounding_box']['min_y'] > timestamps[current_timestamp_index + 1]['bounding_box']['min_y']:
                    current_timestamp_index += 1

                current_timestamp_text = timestamps[current_timestamp_index]['text']
                # 将文本时间戳转换为datetime对象
                current_timestamp = ImageTools.parse_timestamp(
                    current_timestamp_text)

                # 如果是中间的时间戳，为上面的消息调整时间
                if current_timestamp_index > 0 and \
                        message['bounding_box']['min_y'] < timestamps[current_timestamp_index]['bounding_box']['min_y']:
                    current_timestamp -= timedelta(minutes=10)

                message['timestamp'] = current_timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S")
                last_assigned_timestamp = current_timestamp
        else:
            # 如果没有时间戳，留空处理
            for message in messages:
                message['timestamp'] = ""
                # TODO:后续多张图片做滑动窗口平均模拟。

    def process_image(self):
        """Execute the workflow of processing an image."""
        self.extracted_texts = self.extract_text()
        self.detected_objects = self.detect_objects()
        self.assign_labels_to_texts()
        self.merge_texts_by_detection_boxes()
        self.determine_chat_type()
        self.assign_nicknames_to_avatars()
        self.assign_speaker_to_messages()
        self.assign_timestamps_to_messages()
        return self.format_conversations()


    def organize_conversations(self):
        """根据提取的文本和检测到的对象组织对话。"""
        COCO_CLASSES = [
            'avatar', 'message', 'header', 'other', 'transfer', 'file', 'timestamp', 'image',
            'meme', 'comment', 'translate', 'nickname', 'unpassed_message', 'voice', 'withdraw',
        ]
        # 读取图片
        image = cv2.imread(self.image_path)
        image_width = image.shape[1]
        # 提取文本
        extracted_texts = self.extract_text(image)

        # 执行目标检测
        detected_objects = self.detect_objects(
            image, self.config_file, self.checkpoint_file, COCO_CLASSES)
        # 为文本分配属性标签
        self.assign_labels_to_texts(detected_objects, extracted_texts)
        # 合并连续文本
        extracted_texts = self.merge_texts_by_detection_boxes(
            extracted_texts, detected_objects)
        
        # 确定对话类型和名称
        dialog_type, dialog_name = self.determine_chat_type(extracted_texts, image, [
                                                    obj['bbox'] for obj in detected_objects if obj['label'] == 'avatar'])
        
        avatars = [
            obj for obj in detected_objects if obj['label'] == 'avatar']  # 找到所有头像
        # 为头像分配昵称
        avatars_with_nicknames = self.assign_nicknames_to_avatars(
            extracted_texts, avatars, dialog_type, image)

        # 检查每个avatar是否都被分配了昵称，
        
        # 为消息分配说话人
        self.assign_speaker_to_messages(
            extracted_texts, avatars_with_nicknames, dialog_type, dialog_name, image_width)

        # 为消息分配时间戳
        self.assign_timestamps_to_messages(extracted_texts)

        # 组织对话数据
        conversations = []
        for text_info in extracted_texts:
            if text_info['assigned_label'] == 'message':
                conversations.append({
                    "timestamp": text_info.get('timestamp', ""),
                    "speaker": text_info.get('speaker', "未知"),
                    "content": text_info['text'],
                    "message_bbox": text_info['bounding_box'],
                    "image": "",
                    "transfer": [],
                    "file": []
                })

        # 构建最终的结构化数据
        structured_data = {
            "dialog_name": dialog_name,
            "conversation": conversations
        }

        return structured_data



def main():
    parser = argparse.ArgumentParser(
        description='Process chat images for structured data extraction.')
    parser.add_argument('--image_path', type=str,
                        default='/root/wangqun/layoutocr_magic_llava/Batch/5a0e7c5b4eacab366851146d_image_0.jpg', help='Path to the image file')
    parser.add_argument('--config_file', type=str,
                        default='tool/mmdetection/work_dirs/screen_250_data_augment/detr_r50_8xb2-150e_250_screenshot_coco_copy.py', help='Path to the model config file')
    parser.add_argument('--checkpoint_file', type=str,
                        default='tool/mmdetection/work_dirs/screen_250_data_augment/best_coco_bbox_mAP_epoch_149.pth', help='Path to the model checkpoint file')

    args = parser.parse_args()

    chat_organizer = ChatOrganizer(
        args.image_path, args.config_file, args.checkpoint_file)
    structured_data = chat_organizer.organize_conversations()
    print(json.dumps(structured_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
