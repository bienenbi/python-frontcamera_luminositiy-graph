�@���̃v���O�����́ASurface Pro 4�̃t�����g�J��������摜���擾��,
���̉摜�̋P�x�l���O���t�Ƃ��ăv���b�g����v���O�����ł���.
  �v���O�������N������ƃt�����g�J��������摜�����o����. ������,��xQ�������Ƃ��̎��_�܂ł̃O���t��
�v���b�g��,����������Q�������ƏI������.

##############################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture(1)        		// Surface Pro 4 �̃t�����g�J������p����ꍇ��2�ɂ��邱��.
x = np.array([])
y = np.array([])

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)				// �J�������B�e���Ă���摜���f���R�[�h.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)	// �J�����摜��RGB����O���[�X�P�[���摜�ɂ���R�[�h.
    gray_mean = np.mean(gray)				// �O���[�̉摜�̋P�x�𕽋ς���R�[�h.
    y = np.append(y, gray_mean)				// �P�x��z��Ɏ��߂�R�[�h.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot(y)						// ���while�����I��������,�z��Ɏ��߂�ꂽ�P�x���O���t������.
plt.show()
video_capture.release()
cv2.destroyAllWindows()

#####################################################################################

	�Q�l�ɂ����T�C�g

http://tadaoyamaoka.hatenablog.com/entry/2017/02/21/201919 (�ŏI�{�����@2019/07/19)