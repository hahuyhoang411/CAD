#=========IMPORT LIBRARY===========
from typing import Optional, Any

import streamlit as st
st.set_page_config(layout="wide")
maxUploadSize = 200
#with st.cache(allow_output_mutation=True):

#library
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix,log_loss,plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import base64
#model
import xgboost as xgb
from xgboost import XGBClassifier

#===SET UP 3 PAGES SELECTION===


#SIDEBAR
new_title = '<p style="font-size: 50px">MeptiC</p>'
st.sidebar.markdown(new_title, unsafe_allow_html=True)
st.sidebar.markdown('Phần mềm tiên lượng kì vọng sống và thuốc phù hợp cho bệnh nhân mắc bệnh tim thiếu máu cục bộ')
st.sidebar.title("-------------------------------")
page = st.sidebar.radio("Chuyển tới:", options = ['Giới thiệu','Nhập số liệu và dự đoán','Các khuyến cáo'],key='1')

if page == 'Giới thiệu': #PAGE 1
    #===========NAME============
    #st.title('MeptiC')

    #GIF
    #file_ = open("MeptiC.gif", "rb")
    #contents = file_.read()
    #data_url = base64.b64encode(contents).decode("utf-8")
    #file_.close()

    #st.markdown(
    #    f'<img src="data:image/gif;base64,{data_url}" alt="test_medical">',
    #    unsafe_allow_html=True,
    #)

    ###VIDEO
    #video_file = open('MeptiC.mp4', 'rb')
    #video_bytes = video_file.read()

    #st.video(video_bytes, start_time=0)
    ###IMAGE
    #image = Image.open('MeptiC.jpg')

    #st.image(image,use_column_width=True)

    #BACKGROUND
    main_bg = "MeptiC.jpg"
    main_bg_ext = "jpg"

    #side_bg = "sample.jpg"
    #side_bg_ext = "jpg"

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if page == 'Nhập số liệu và dự đoán': #PAGE 2
    # ==========Get info============
    # header
    st.markdown("""
    <style>
    .big-font {
        font-size:60px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Thông tin bệnh nhân</p>', unsafe_allow_html=True)
    st.success('Trang để nhập các thông tin và xét nghiệm của bệnh nhân')

#=======PAGE 2=======
#====DEMOGRAPHIC====
    #PART 1 LABELS
    uploaded_file = st.sidebar.file_uploader("Đưa file .csv của bạn vào đây:", type=["csv"],help='Hãy chuyển về đuôi .csv')
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.title('**Dữ liệu nhập tay**')
            with st.container():
                col1, col2= st.columns([1, 1])

            col1.header('_Nhân khẩu học_')
            col2.header('_Chẩn đoán và Thủ thuật_')

            #ELEMENT PART1
            with st.container():
                demo, demo2, diag = st.columns([1,1,2])

            #DEMO
            with st.container():
                gender = demo.selectbox('Giới tính:',
                         ('Nam','Nữ'))
                gender = 0 if gender == 'Nữ' else 1 #gender change
                age = demo.slider('Tuổi:', 18, 89)
                ethi = demo.selectbox('Dân tộc:',
                                ('Da vàng', 'Da trắng', 'Da đen',
                                 'Mỹ bản địa', 'Mỹ Latinh', 'Trung Đông', 'Khác'))
                #ethinity
                if ethi == 'Da trắng':
                    ethi = 6
                elif ethi == 'Khác':
                    ethi = 5
                elif ethi == 'Mỹ Latinh':
                    ethi = 3
                elif ethi == 'Da đen':
                    ethi = 2
                elif ethi == 'Da vàng':
                    ethi = 1
                elif ethi == 'Mỹ bản địa':
                    ethi = 0
                elif ethi == 'Trung Đông':
                    ethi = 4

                marry = demo.selectbox('Tình trạng hôn nhân:',
                                 ('Độc thân', 'Đã kết hôn', 'Góa phụ',
                                  'Ly dị', 'Li thân', 'Bạn đời', 'Không biết'))
                #marrital
                if marry == 'Độc thân':
                    marry = 4
                elif marry == 'Đã kết hôn':
                    marry = 2
                elif marry == 'Góa phụ':
                    marry = 6
                elif marry == 'Ly dị':
                    marry = 0
                elif marry == 'Li thân':
                    marry = 3
                elif marry == 'Bạn đời':
                    marry = 1
                elif marry == 'Không biết':
                    marry = 5

            with st.container():
                smoke = demo2.selectbox('Hút thuốc:',
                                       ('Có', 'Không'))
                smoke = 0 if smoke == 'Không' else 1  # smoke change
                surgery = demo2.selectbox('Tình trạng phẫu thuật:',
                                   ('Có','Không'))
                surgery = 0 if surgery == 'Không' else 1  # surgery change
                bmi = demo2.number_input('Chỉ số BMI:', value = 20)
                stay = demo2.number_input('Lần nhấp viện thứ:',step=1, value = 1)

            #DIAG AND PROCE
            with st.container():
                option = diag.multiselect('Các bệnh của bệnh nhân:',
                                        ['(41001) Nhồi máu cơ tim cấp của thành trước bên, giai đoạn chăm sóc ban đầu',
                                        '(41011) Nhồi máu cơ tim cấp của thành trước khác, giai đoạn chăm sóc ban đầu',
                                        '(41021) Nhồi máu cơ tim cấp tính của thành bên, giai đoạn chăm sóc ban đầu',
                                        '(41031) Nhồi máu cơ tim cấp của thành dưới, giai đoạn chăm sóc ban đầu',
                                        '(41041) Nhồi máu cơ tim cấp của thành dưới khác, giai đoạn chăm sóc ban đầu',
                                        '(41071) Nhồi máu cơ tim, giai đoạn chăm sóc ban đầu',
                                        '(41072) Nhồi máu cơ tim, giai đoạn chăm sóc tiếp theo',
                                        '(41081) Nhồi máu cơ tim cấp tính tại các vị trí được chỉ định khác, giai đoạn chăm sóc ban đầu',
                                        '(41091) Nhồi máu cơ tim cấp tính ở vị trí không xác định',
                                        '(4111) Hội chứng mạch vành trung gian',
                                        '(41189) Các dạng bệnh tim thiếu máu cục bộ cấp tính và bán cấp tính khác',
                                        '(4139) Đau thắt ngực khác và không xác định',
                                        '(41400) Xơ vữa động mạch thược loại mạch không xác định, tự nhiên hoặc ghép',
                                        '(41401) Xơ vữa động mạch vành nguyên phát',
                                        '(41402) Xơ vữa động mạch vành của ghép nối động mạch tự thông',
                                        '(41412) Phình mạch vành',
                                        '(4142) Tắc hoàn toàn mãn tính của động mạch vành',
                                        '(4148) Các dạng bệnh tim thiếu máu cục bộ mạn tính được chỉ định khác'])

                DIAG_4111, DIAG_4139, DIAG_4142, DIAG_4148, DIAG_41001, DIAG_41011, DIAG_41021, DIAG_41031, DIAG_41041, DIAG_41071, DIAG_41072, DIAG_41081, DIAG_41091, DIAG_41189, DIAG_41400, DIAG_41401, DIAG_41402, DIAG_41412 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                if '(4111) Hội chứng mạch vành trung gian' in option: #1
                    DIAG_4111 = 1
                if '(4139) Đau thắt ngực khác và không xác định'in option:#2
                    DIAG_4139 = 1
                if '(4142) Tắc hoàn toàn mãn tính của động mạch vành'in option:#3
                    DIAG_4142 = 1
                if '(4148) Các dạng bệnh tim thiếu máu cục bộ mạn tính được chỉ định khác'in option:#4
                    DIAG_4148 = 1
                if '(41001) Nhồi máu cơ tim cấp của thành trước bên, giai đoạn chăm sóc ban đầu'in option:#5
                    DIAG_41001 = 1
                if '(41011) Nhồi máu cơ tim cấp của thành trước khác, giai đoạn chăm sóc ban đầu' in option: #6
                    DIAG_41011 = 1
                if '(41021) Nhồi máu cơ tim cấp tính của thành bên, giai đoạn chăm sóc ban đầu'in option:#7
                    DIAG_41021 = 1
                if '(41031) Nhồi máu cơ tim cấp của thành dưới, giai đoạn chăm sóc ban đầu'in option:#8
                    DIAG_41031 = 1
                if '(41041) Nhồi máu cơ tim cấp của thành dưới khác, giai đoạn chăm sóc ban đầu'in option:#9
                    DIAG_41041 = 1
                if '(41071) Nhồi máu cơ tim, giai đoạn chăm sóc ban đầu'in option:#10
                    DIAG_41071 = 1
                if '(41072) Nhồi máu cơ tim, giai đoạn chăm sóc tiếp theo' in option: #11
                    DIAG_41072 = 1
                if '(41081) Nhồi máu cơ tim cấp tính tại các vị trí được chỉ định khác, giai đoạn chăm sóc ban đầu'in option:#12
                    DIAG_41081 = 1
                if '(41091) Nhồi máu cơ tim cấp tính ở vị trí không xác định'in option:#13
                    DIAG_41091 = 1
                if '(41189) Các dạng bệnh tim thiếu máu cục bộ cấp tính và bán cấp tính khác'in option:#14
                    DIAG_41189 = 1
                if '(41400) Xơ vữa động mạch thược loại mạch không xác định, tự nhiên hoặc ghép'in option:#15
                    DIAG_41400 = 1
                if '(41401) Xơ vữa động mạch vành nguyên phát' in option: #16
                    DIAG_41401 = 1
                if '(41402) Xơ vữa động mạch vành của ghép nối động mạch tự thông'in option:#17
                    DIAG_41402 = 1
                if '(41412) Phình mạch vành'in option:#18
                    DIAG_41412 = 1



            with st.container():
                option_pro = diag.multiselect('Các thủ thuật thực hiện:',
                                              ['(51) Cấy máy khử rung tim tái đồng bộ, hệ thống tổng thể [CRT-D]',
                                               '(54)Chỉ cấy hoặc thay thế máy phát xung máy khử rung tim tái đồng bộ hóa tim [CRT-D]',
                                               '(66)Nong mạch vành qua da [PTCA]',
                                               '(3603)Nong động mạch vành ngực hở',
                                               '(3606) Đặt (các) stent động mạch vành không dùng thuốc rửa giải',
                                               '(3607) Đặt (các) stent động mạch vành rửa giải bằng thuốc',
                                               '(3611)(Động mạch chủ) bắc cầu mạch vành của một động mạch vành',
                                               '(3612) (Động mạch chủ) bắc cầu mạch vành của hai động mạch vành',
                                               '(3613)(Động mạch chủ) bắc cầu mạch vành của ba động mạch vành',
                                               '(3614)(Động mạch chủ) bắc cầu mạch vành của bốn hoặc nhiều động mạch vành',
                                               '(3615) Cầu nối động mạch vành-động mạch vú đơn bên trong',
                                               '(3616) Cầu nối động mạch vành - động mạch vú đôi bên trong',
                                               '(3617) Bắc cầu động mạch vành bụng',
                                               '(3721) Thông tim phải',
                                               '(3722) Thông tim trái',
                                               '(3723) Kết hợp thông tim phải và tim trái',
                                               '(3768) Lắp thiết bị trợ tim ngoài qua da',
                                               '(3778) Lắp đặt hệ thống máy tạo nhịp tim truyền tĩnh mạch tạm thời',
                                               '(3795) Chỉ cấy(các) dây dẫn máy khử rung tim / máy khử rung tim tự động',
                                               '(3796) Chỉ cấy máy phát xung máy khử rung tim / máy khử rung tim tự động',
                                               '(3797) Chỉ thay thế(các) dây dẫn máy khử rung tim / máy khử rung tim tự động',
                                               '(3798) Chỉ thay thế máy phát xung máy khử rung tim / máy khử rung tim tự động',
                                               '(3964) Máy tạo nhịp tim trong phẫu thuật',
                                               '(8855) Chụp động mạch vành bằng một ống thông duy nhất',
                                               '(8856) Chụp động mạch vành bằng hai ống thông',
                                               ])

                PROCE_51, PROCE_54, PROCE_66, PROCE_3603, PROCE_3606, PROCE_3607, PROCE_3611, PROCE_3612, PROCE_3613, PROCE_3722, PROCE_3614, PROCE_3615, PROCE_3616, PROCE_3617, PROCE_3721, PROCE_3723, PROCE_3768, PROCE_3778, PROCE_3795, PROCE_3796, PROCE_3797, PROCE_3798, PROCE_3964, PROCE_8855, PROCE_8856 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                if '(51) Cấy máy khử rung tim tái đồng bộ, hệ thống tổng thể [CRT-D]' in option_pro: #1
                    PROCE_51 = 1
                if '(54)Chỉ cấy hoặc thay thế máy phát xung máy khử rung tim tái đồng bộ hóa tim [CRT-D]'in option_pro:#2
                    PROCE_54 = 1
                if '(66)Nong mạch vành qua da [PTCA]'in option_pro:#3
                    PROCE_66 = 1
                if '(3603)Nong động mạch vành ngực hở'in option_pro:#4
                    PROCE_3603 = 1
                if '(3606) Đặt (các) stent động mạch vành không dùng thuốc rửa giải'in option_pro:#5
                    PROCE_3606 = 1
                if '(3607) Đặt (các) stent động mạch vành rửa giải bằng thuốc]'in option_pro:#6
                    PROCE_3607 = 1
                if '(3611)(Động mạch chủ) bắc cầu mạch vành của một động mạch vành'in option_pro:#7
                    PROCE_3611 = 1
                if '(3612) (Động mạch chủ) bắc cầu mạch vành của hai động mạch vành'in option_pro:#8
                    PROCE_3612 = 1
                if '(3613)(Động mạch chủ) bắc cầu mạch vành của ba động mạch vành'in option_pro:#9
                    PROCE_3613 = 1
                if '(3722) Thông tim trái'in option_pro:#10
                    PROCE_3722 = 1
                if '(3614)(Động mạch chủ) bắc cầu mạch vành của bốn hoặc nhiều động mạch vành'in option_pro:#11
                    PROCE_3614 = 1
                if '(3615) Cầu nối động mạch vành-động mạch vú đơn bên trong' in option_pro:#12
                    PROCE_3615 = 1
                if '(3616) Cầu nối động mạch vành - động mạch vú đôi bên trong' in option_pro:#13
                    PROCE_3616 = 1
                if '(3617) Bắc cầu động mạch vành bụng' in option_pro:#14
                    PROCE_3617 = 1
                if '(3721) Thông tim phải' in option_pro:#15
                    PROCE_3721 = 1
                if '(3723) Kết hợp thông tim phải và tim trái'in option_pro:#16
                    PROCE_3723 = 1
                if '(3768) Lắp thiết bị trợ tim ngoài qua da'in option_pro:#17
                    PROCE_3768 = 1
                if '(3778) Lắp đặt hệ thống máy tạo nhịp tim truyền tĩnh mạch tạm thời'in option_pro:#18
                    PROCE_3778 = 1
                if '(3795) Chỉ cấy(các) dây dẫn máy khử rung tim / máy khử rung tim tự động'in option_pro:#19
                    PROCE_3795 = 1
                if '(3796) Chỉ cấy máy phát xung máy khử rung tim / máy khử rung tim tự động'in option_pro:#20
                    PROCE_3796 = 1
                if '(3797) Chỉ thay thế(các) dây dẫn máy khử rung tim / máy khử rung tim tự động'in option_pro:#21
                    PROCE_3797 = 1
                if '(3798) Chỉ thay thế máy phát xung máy khử rung tim / máy khử rung tim tự động'in option_pro:#22
                    PROCE_3798 = 1
                if '(3964) Máy tạo nhịp tim trong phẫu thuật'in option_pro:#23
                    PROCE_3964 = 1
                if '(8855) Chụp động mạch vành bằng một ống thông duy nhất'in option_pro:#24
                    PROCE_8855 = 1
                if '(8856) Chụp động mạch vành bằng hai ống thông' in option_pro: #25
                    PROCE_8856 = 1

                # ===PART2===
                with st.expander("Chỉ số xét nghiệm"):

                    vit_head, lab_head = st.columns([1, 1])
                    vit_head.header('Chỉ số thông thường')
                    lab_head.header('Chỉ số đặc trưng')

                    vital1, vital2, vital3, lab = st.columns([1, 1, 1, 1])

                    heartrate_min = vital1.number_input('Nhịp tim nhỏ nhất (bpm):')
                    sysbp_min = vital1.number_input('Huyết áp nhỏ nhất (mmHg):')
                    diasbp_min = vital1.number_input('Huyết áp tâm thu nhỏ nhất (mmHg):')
                    meanbp_min = vital1.number_input('Huyết áp tâm trương nhỏ nhất (mmHg):')
                    resprate_min = vital1.number_input('Nhịp thở nhỏ nhất (bpm):')
                    tempc_min = vital1.number_input('Nhiệt độ nhỏ nhất (C):')
                    spo2_min = vital1.number_input('SpO2 nhỏ nhất (%):')
                    glucose_min = vital1.number_input('Glucose nhỏ nhất (mg/dL):')

                    heartrate_max = vital2.number_input('Nhịp tim lớn nhất (bpm):')
                    sysbp_max = vital2.number_input('Huyết áp lớn nhất (mmHg):')
                    diasbp_max = vital2.number_input('Huyết áp tâm thu lớn nhất (mmHg):')
                    meanbp_max = vital2.number_input('Huyết áp tâm trương lớn nhất (mmHg):')
                    resprate_max = vital2.number_input('Nhịp thở lớn nhất (bpm):')
                    tempc_max = vital2.number_input('Nhiệt độ lớn nhất (C):')
                    spo2_max = vital2.number_input('SpO2 lớn nhất (%):')
                    glucose_max = vital2.number_input('Glucose lớn nhất (mg/dL):')

                    heartrate_mean = vital3.number_input('Nhịp tim trung bình (bpm):')
                    sysbp_mean = vital3.number_input('Huyết áp trung bình (mmHg):')
                    diasbp_mean = vital3.number_input('Huyết áp tâm thu trung bình (mmHg):')
                    meanbp_mean = vital3.number_input('Huyết áp tâm trương trung bình (mmHg):')
                    resprate_mean = vital3.number_input('Nhịp thở trung bình (bpm):')
                    tempc_mean = vital3.number_input('Nhiệt độ trung bình (C):')
                    spo2_mean = vital3.number_input('SpO2 trung bình (%):')
                    glucose_mean = vital3.number_input('Glucose trung bình (mg/dL):')

                    # LAB_FEATURE
                    choles = lab.number_input('Tỉ lệ cholesterol:', value = 1.00)
                    choles_total = lab.number_input('Tổng lượng cholesterol (mg/dL):',value = 150.00)
                    hdl = lab.number_input('HDL (mg/dL):', value = 60.00)
                    ldl = lab.number_input('LDL (mg/dL):', value = 70.00)
                    tri = lab.number_input('Triglycerids (mg/dL):', value = 120.00)
                    tro = lab.number_input('TroponinT (ng/L):', value = 10.00)
                    ck_mb = lab.number_input('Ckmb (u/L):', value = 20.00)
                    ck = lab.number_input('Ck (u/L):', value = 80.00)

        #===SCORE PIVLAB===
            with st.expander("Chỉ số sinh hóa và thang điểm"):

                piv_head, score_head = st.columns([1, 1])
                piv_head.header('Chỉ số sinh hóa')
                score_head.header("Thang điểm")

                piv, piv2, score, score2 = st.columns([1, 1, 1, 1])

                ev1 = score.number_input('Điểm elix vanwalraven:')
                ev2 = score.number_input('Điểm elix sid29:')
                ev3 = score.number_input('Điểm elix sid30')
                gcs1 = score.number_input('Điểm mingcs:')
                gcs2 = score.number_input('Điểm gcsmotor:')
                gcs3 = score.number_input('Điểm gcsverbal:')
                gcs4 = score.number_input('Điểm gcseyes:')

                endo = score2.number_input('Điểm endotrachflag:')
                oasis = score2.number_input('Điểm OASIS:')
                sofa = score2.number_input('Điểm SOFA:')
                saps = score2.number_input('Điểm SAPS:')

                albumin = piv.number_input('Albumin (g/L):', value = 40.00)
                anion = piv.number_input('Aninongap (mEq/l):', value = 10.00)
                bicar = piv.number_input('Bicarbonate (mmol/l):', value = 22.00)
                bili = piv.number_input('Bilirubin (mg/dL):', value = 0.50)
                crea = piv.number_input('Creatinine (mg/dL):', value = 1.00)
                chlo = piv.number_input('Chloride (mEq/L):', value = 100.00)
                glu = piv.number_input('Glucose (mmol/L):', value = 120.00)
                hema = piv.number_input('Hematocrit (L/L):', value = 0.40)
                hemo = piv.number_input('Hemoglobin (g/dL):', value = 13.00)
                lac = piv2.number_input('Lactate (mg/dL):', value = 10.00)
                Plate = piv2.number_input('Platelet (g/L):', value = 200.00)
                potassium = piv2.number_input('Kali (mEq/L):', value = 4.00)
                ptt = piv2.number_input('aPTT (s):', value = 25.00)
                inr = piv2.number_input('INR:', value = 1.00)
                pt = piv2.number_input('PT (s):', value = 10.00)
                sodium = piv2.number_input('Natri (mmol/L):', value = 140.00)
                bun = piv2.number_input('BUN (mg/dL):', value = 10.00)
                wbc = piv2.number_input('WBC (g/L):', value = 5.00)

                #NOT DEFINE YET
                los_hos = 0
                first_hosp_stay = 0
                los_icu = 0
                first_icu_stay = 0
                first_icu_stay_demo = 0
                seq_num = 0


#DRUGS USED OR USING
            with st.container():
                st.header('_Thuốc bệnh nhân đã và đang sử dụng_')

                plate, coagu, fibrin, lipid = st.columns([1, 1, 1, 1])
                plate.subheader('Chống kết tập tiểu cầu')
                coagu.subheader('Chống đông máu')
                lipid.subheader('Trị lipid huyết')
                fibrin.subheader('Li giải fibrin')
                #DRUG_B01AA03, DRUG_B01AB01, DRUG_B01AB05, DRUG_B01AC04, DRUG_B01AC06, DRUG_B01AC16, DRUG_B01AC22, DRUG_B01AD02, DRUG_B01AE03, DRUG_B01AE06, DRUG_C01DA02, DRUG_C01DA08, DRUG_C01DA14, DRUG_C01EB18, DRUG_C07AB02, DRUG_C07AB09, DRUG_C07AG01, DRUG_C07AG02, DRUG_C07CB02, DRUG_C08CA01, DRUG_C08CA04, DRUG_C08DA01, DRUG_C08DB01, DRUG_C09AA01, DRUG_C09AA03, DRUG_C09AA05, DRUG_C09AA13, DRUG_C09BA02, DRUG_C10AA01, DRUG_C10AA03, DRUG_C10AA05, DRUG_C10AA07, DRUG_C10AB04, DRUG_C10AD02, DRUG_C10AX06, DRUG_C10AX09, DRUG_N02BE51 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                #def true_check(self):
                #    self = 0 if self == False else 1

                    # Part1
                DRUG_B01AC06 = plate.checkbox('Aspirin')
                DRUG_B01AC06 = 0 if DRUG_B01AC06 == False else 1

                DRUG_N02BE51 = plate.checkbox('Excedrin')
                DRUG_N02BE51 = 0 if DRUG_N02BE51 == False else 1

                DRUG_B01AC04 = plate.checkbox('Clopidogrel')
                DRUG_B01AC04 = 0 if DRUG_B01AC04 == False else 1

                DRUG_B01AC16 = plate.checkbox('Eptifibatide')
                DRUG_B01AC16 = 0 if DRUG_B01AC16 == False else 1

                DRUG_B01AC22 = plate.checkbox('Prasugrel')
                DRUG_B01AC22 = 0 if DRUG_B01AC22 == False else 1


                DRUG_B01AB01 = coagu.checkbox('Heparin')
                DRUG_B01AB01 = 0 if DRUG_B01AB01 == False else 1

                DRUG_B01AB05 = coagu.checkbox('Enoxaheparin')
                DRUG_B01AB05 = 0 if DRUG_B01AB05 == False else 1

                DRUG_B01AA03 = coagu.checkbox('Warfarin')
                DRUG_B01AA03 = 0 if DRUG_B01AA03 == False else 1

                DRUG_B01AE03 = coagu.checkbox('Argatroban')
                DRUG_B01AE03 = 0 if DRUG_B01AE03 == False else 1

                DRUG_B01AE06 = coagu.checkbox('Bivalirudin')
                DRUG_B01AE06 = 0 if DRUG_B01AE06 == False else 1



                strep = fibrin.checkbox('Streptokinase')
                DRUG_B01AD02 = fibrin.checkbox("Alteplase")
                DRUG_B01AD02 = 0 if DRUG_B01AD02 == False else 1

                fibrin.subheader('Nitrat hữu cơ')

                DRUG_C01DA02 = fibrin.checkbox('Nitroglycerin')
                DRUG_C01DA02 = 0 if DRUG_C01DA02 == False else 1

                DRUG_C01DA08 = fibrin.checkbox('Isosorbide')
                DRUG_C01DA08 = 0 if DRUG_C01DA08 == False else 1

                DRUG_C01DA14 = fibrin.checkbox('Isosorbide Dinitrate')
                DRUG_C01DA14 = 0 if DRUG_C01DA14 == False else 1

                DRUG_C10AA01 = lipid.checkbox('Simvastatin')
                DRUG_C10AA01 = 0 if DRUG_C10AA01 == False else 1

                DRUG_C10AA03 = lipid.checkbox('Pravastatin')
                DRUG_C10AA03 = 0 if DRUG_C10AA03 == False else 1

                DRUG_C10AA05 = lipid.checkbox('Atorvastatin')
                DRUG_C10AA05 = 0 if DRUG_C10AA05 == False else 1

                DRUG_C10AA07 = lipid.checkbox('Rosuvastatin')
                DRUG_C10AA07 = 0 if DRUG_C10AA07 == False else 1

                DRUG_C10AB04 = lipid.checkbox('Gemfibrozil')
                DRUG_C10AB04 = 0 if DRUG_C10AB04 == False else 1

                DRUG_C10AD02 = lipid.checkbox('Niacin')
                DRUG_C10AD02 = 0 if DRUG_C10AD02 == False else 1

                DRUG_C10AX06 = lipid.checkbox('Dầu cá (Omega 3)')
                DRUG_C10AX06 = 0 if DRUG_C10AX06 == False else 1

                DRUG_C10AX09 = lipid.checkbox('Ezetimibe')
                DRUG_C10AX09 = 0 if DRUG_C10AX09 == False else 1


                beta, acei, calci, natri = st.columns([1, 1, 1, 1])
                natri.subheader('Chẹn kênh Natri')
                beta.subheader('Chẹn Beta')
                calci.subheader('Chẹn kênh Calci')
                acei.subheader('Ức chế men chuyển')

                DRUG_C01EB18 = natri.checkbox('Ranolazine')
                DRUG_C01EB18 = 0 if DRUG_C01EB18 == False else 1


                DRUG_C07AB02 = beta.checkbox('Metoprolol')
                DRUG_C07AB02 = 0 if DRUG_C07AB02 == False else 1

                DRUG_C07AB09 = beta.checkbox('Esmolol')
                DRUG_C07AB09 = 0 if DRUG_C07AB09 == False else 1

                DRUG_C07AG01 = beta.checkbox('Labetalol')
                DRUG_C07AG01 = 0 if DRUG_C07AG01 == False else 1

                DRUG_C07AG02 = beta.checkbox('Carvedilol')
                DRUG_C07AG02 = 0 if DRUG_C07AG02 == False else 1

                DRUG_C07CB02 = beta.checkbox('Atenolol')
                DRUG_C07CB02 = 0 if DRUG_C07CB02 == False else 1


                DRUG_C08CA01 = calci.checkbox('Amlodipine')
                DRUG_C08CA01 = 0 if DRUG_C08CA01 == False else 1

                DRUG_C08CA04 = calci.checkbox('Nicardipine')
                DRUG_C08CA04 = 0 if DRUG_C08CA04 == False else 1

                DRUG_C08DA01 = calci.checkbox('Verapamil')
                DRUG_C08DA01 = 0 if DRUG_C08DA01 == False else 1

                DRUG_C08DB01 = calci.checkbox('Diltiazem')
                DRUG_C08DB01 = 0 if DRUG_C08DB01 == False else 1


                DRUG_C09AA01 = acei.checkbox('Captopril')
                DRUG_C09AA01 = 0 if DRUG_C09AA01 == False else 1

                DRUG_C09AA03 = acei.checkbox('Lisinopril')
                DRUG_C09AA03 = 0 if DRUG_C09AA03 == False else 1

                DRUG_C09AA05 = acei.checkbox('Ramipril')
                DRUG_C09AA05 = 0 if DRUG_C09AA05 == False else 1

                DRUG_C09AA13 = acei.checkbox('Moexipril')
                DRUG_C09AA13 = 0 if DRUG_C09AA13 == False else 1

                DRUG_C09BA02 = acei.checkbox('Enalapril')
                DRUG_C09BA02 = 0 if DRUG_C09BA02 == False else 1


            data = {'DEMO_age': age,
                    'DEMO_bmi': bmi,
                    'DEMO_gender': gender,
                    'DEMO_marital_status': marry,
                    'DEMO_ethnicity': ethi,
                    'DEMO_los_hospital': los_hos,
                    'DEMO_hospstay_seq': stay,
                    'DEMO_first_hosp_stay': first_hosp_stay,
                    'ICU_los_icu': los_icu,
                    'ICU_first_icu_stay': first_icu_stay,
                    'DEMO_first_icu_stay': first_icu_stay_demo,
                    "DIAG_seq_num": seq_num,
                    'ICU_SURGERY_FLAG': surgery,
                    'DIAG_4111': DIAG_4111,
                    'DIAG_4139': DIAG_4139,
                    'DIAG_4142': DIAG_4142,
                    'DIAG_4148':  DIAG_4148,
                    'DIAG_41001': DIAG_41001,
                    'DIAG_41011': DIAG_41011,
                    'DIAG_41021': DIAG_41021,
                    'DIAG_41031': DIAG_41031,
                    'DIAG_41041': DIAG_41041,
                    'DIAG_41071': DIAG_41071,
                    'DIAG_41072': DIAG_41072,
                    'DIAG_41081': DIAG_41081,
                    'DIAG_41091': DIAG_41091,
                    'DIAG_41189': DIAG_41189,
                    'DIAG_41400': DIAG_41400,
                    'DIAG_41401': DIAG_41401,
                    'DIAG_41402': DIAG_41402,
                    'DIAG_41412': DIAG_41412,
                    'DEMO_smoke': smoke,
                    'SCORE_elixhauser_vanwalraven': ev1,
                    'SCORE_elixhauser_sid29': ev2,
                    'SCORE_elixhauser_sid30': ev3,
                    'SCORE_mingcs': gcs1,
                    'SCORE_gcsmotor': gcs2,
                    'SCORE_gcsverbal': gcs3,
                    'SCORE_gcseyes': gcs4,
                    'SCORE_endotrachflag': endo,
                    'SCORE_oasis_avg': oasis,
                    'SCORE_sofa_avg': sofa,
                    'SCORE_saps_avg': saps,
                    'LABF_cholesterol_ratio': choles,
                    'LABF_hdl_cholesterol': hdl,
                    'LABF_ldl_cholesterol': ldl,
                    'LABF_total_cholesterol': choles_total,
                    'LABF_triglycerids': tri,
                    'LABF_troponint': tro,
                    'LABF_ck_mb': ck_mb,
                    'LABF_ck': ck,
                    'PIVLAB_aniongap': anion,
                    'PIVLAB_albumin' : albumin,
                    'PIVLAB_bicarbonate': bicar,
                    'PIVLAB_bilirubin': bili,
                    'PIVLAB_creatinine': crea,
                    'PIVLAB_chloride': chlo,
                    'PIVLAB_glucose'  : glu,
                    'PIVLAB_hematocrit': hema,
                    'PIVLAB_hemoglobin': hemo,
                    'PIVLAB_lactate': lac,
                    'PIVLAB_platelet': Plate,
                    'PIVLAB_potassium': potassium,
                    'PIVLAB_ptt': ptt,
                    'PIVLAB_inr': inr,
                    'PIVLAB_pt': pt,
                    'PIVLAB_sodium': sodium,
                    'PIVLAB_bun': bun,
                    'PIVLAB_wbc': wbc,
                    'VITALFDAY_heartrate_min': heartrate_min,
                    'VITALFDAY_heartrate_max': heartrate_max,
                    'VITALFDAY_heartrate_mean': heartrate_mean,
                    'VITALFDAY_sysbp_min': sysbp_min,
                    'VITALFDAY_sysbp_max': sysbp_max,
                    'VITALFDAY_sysbp_mean': sysbp_mean,
                    'VITALFDAY_diasbp_min': diasbp_min,
                    'VITALFDAY_diasbp_max': diasbp_max,
                    'VITALFDAY_diasbp_mean': diasbp_mean,
                    'VITALFDAY_meanbp_min': meanbp_min,
                    'VITALFDAY_meanbp_max': meanbp_max,
                    'VITALFDAY_meanbp_mean': meanbp_mean,
                    'VITALFDAY_resprate_min': resprate_min,
                    'VITALFDAY_resprate_max': resprate_max,
                    'VITALFDAY_resprate_mean': resprate_mean,
                    'VITALFDAY_tempc_min': tempc_min,
                    'VITALFDAY_tempc_max': tempc_max,
                    'VITALFDAY_tempc_mean': tempc_mean,
                    'VITALFDAY_spo2_min': spo2_min,
                    'VITALFDAY_spo2_max': spo2_max,
                    'VITALFDAY_spo2_mean': spo2_mean,
                    'VITALFDAY_glucose_min': glucose_min,
                    'VITALFDAY_glucose_max': glucose_max,
                    'VITALFDAY_glucose_mean': glucose_mean,
                    'PROCE_51': PROCE_51,
                    'PROCE_54': PROCE_54,
                    'PROCE_66': PROCE_66,
                    'PROCE_3603': PROCE_3603,
                    'PROCE_3606': PROCE_3606,
                    'PROCE_3607': PROCE_3607,
                    'PROCE_3611': PROCE_3611,
                    'PROCE_3612': PROCE_3612,
                    'PROCE_3613': PROCE_3613,
                    'PROCE_3614': PROCE_3614,
                    'PROCE_3615': PROCE_3615,
                    'PROCE_3616': PROCE_3616,
                    'PROCE_3617': PROCE_3617,
                    'PROCE_3721': PROCE_3721,
                    'PROCE_3722': PROCE_3722,
                    'PROCE_3723': PROCE_3723,
                    'PROCE_3768': PROCE_3768,
                    'PROCE_3778': PROCE_3778,
                    'PROCE_3795': PROCE_3795,
                    'PROCE_3796': PROCE_3796,
                    'PROCE_3797': PROCE_3797,
                    'PROCE_3798': PROCE_3798,
                    'PROCE_3964': PROCE_3964,
                    'PROCE_8855': PROCE_8855,
                    'PROCE_8856': PROCE_8856,
                    'DRUG_B01AA03':DRUG_B01AA03,
                    'DRUG_B01AB01':DRUG_B01AB01,
                    'DRUG_B01AB05':DRUG_B01AB05,
                    'DRUG_B01AC04':DRUG_B01AC04,
                    'DRUG_B01AC06':DRUG_B01AC06,
                    'DRUG_B01AC16':DRUG_B01AC16,
                    'DRUG_B01AC22':DRUG_B01AC22,
                    'DRUG_B01AD02':DRUG_B01AD02,
                    'DRUG_B01AE03':DRUG_B01AE03,
                    'DRUG_B01AE06':DRUG_B01AE06,
                    'DRUG_C01DA02':DRUG_C01DA02,
                    'DRUG_C01DA08':DRUG_C01DA08,
                    'DRUG_C01DA14':DRUG_C01DA14,
                    'DRUG_C01EB18':DRUG_C01EB18,
                    'DRUG_C07AB02':DRUG_C07AB02,
                    'DRUG_C07AB09':DRUG_C07AB09,
                    'DRUG_C07AG01':DRUG_C07AG01,
                    'DRUG_C07AG02':DRUG_C07AG02,
                    'DRUG_C07CB02':DRUG_C07CB02,
                    'DRUG_C08CA01':DRUG_C08CA01,
                    'DRUG_C08CA04':DRUG_C08CA04,
                    'DRUG_C08DA01':DRUG_C08DA01,
                    'DRUG_C08DB01':DRUG_C08DB01,
                    'DRUG_C09AA01':DRUG_C09AA01,
                    'DRUG_C09AA03':DRUG_C09AA03,
                    'DRUG_C09AA05':DRUG_C09AA05,
                    'DRUG_C09AA13':DRUG_C09AA13,
                    'DRUG_C09BA02':DRUG_C09BA02,
                    'DRUG_C10AA01':DRUG_C10AA01,
                    'DRUG_C10AA03':DRUG_C10AA03,
                    'DRUG_C10AA05':DRUG_C10AA05,
                    'DRUG_C10AA07':DRUG_C10AA07,
                    'DRUG_C10AB04':DRUG_C10AB04,
                    'DRUG_C10AD02':DRUG_C10AD02,
                    'DRUG_C10AX06':DRUG_C10AX06,
                    'DRUG_C10AX09':DRUG_C10AX09,
                    'DRUG_N02BE51':DRUG_N02BE51
                    }

            features = pd.DataFrame(data, index=[0])
            return features

        users_input_df = user_input_features()

        if uploaded_file is not None:
            st.write(users_input_df)
        else:
            st.subheader('_Đang chờ người dùng nhập file .csv. Hiện tại phần mềm đang sử dụng dữ liệu từ người dùng nhập tay ở trên._')
            st.write(users_input_df)

        # IMPORT DATA
        cad_raw = pd.read_csv('MIMIC3_CAD.csv')
        #DRUG = [c_ for c_ in cad_raw if c_.startswith('DRUG')]  # drug for CAD
        drop = ['hospital_expire_flag'] + ['hadm_id']
        cad_step1 = cad_raw.drop(columns=drop,axis=1)
        df_step1 = pd.concat([users_input_df, cad_step1], axis=0)

        # ENCODE SOME FEATURES (GENDER, MARITAL, ETHINICITY,...?)
        df_step1 = df_step1[:1]

        # LOAD SAVED MODEL
        load_clf = pickle.load(open('step1_clf.pkl', 'rb'))

        # PREDICTION
        prediction = load_clf.predict(df_step1)
        prediction_proba = load_clf.predict_proba(df_step1)

        with st.container():
            st.warning("")
            st.title('Dự đoán tỷ lệ sống của bệnh nhân')
            pred, proba = st.columns([1, 1])

            pred.header('_Tiên lượng của bệnh nhân_')
            if prediction[0] == 0:
                pred.success('Sống')
            else:
                pred.warning('Chết')
            #pred.write(prediction[0])
            proba.header('_Tỷ lệ sống của bệnh nhân_')

            proba.success(f"{str(round(prediction_proba[0][0]*100,2))}%") #%

        #STEP2
        with st.container():
            st.text("")
            st.title('Dự đoán thuốc tối ưu cho bệnh nhân')
            pred2, proba2 = st.columns([1, 1])

            DRUG = [c_ for c_ in cad_raw if c_.startswith('DRUG')]  # drug for CAD
            drop2 = ['hospital_expire_flag'] + ['hadm_id'] + DRUG
            cad_step2 = cad_raw.drop(columns=drop2, axis=1)
            df_step2 = pd.concat([users_input_df, cad_step2], axis=0)
            df_step2 = df_step2.drop(DRUG,axis=1)

        # ENCODE SOME FEATURES (GENDER, MARITAL, ETHINICITY,...?)
            df_step2 = df_step2[:1]

        # LOAD SAVED MODEL
        load_clf2 = pickle.load(open('step2_clf.pkl', 'rb'))

        # PREDICTION
        probas = load_clf2.predict_proba(df_step2)

        #INTERFACE
        probas_array = np.array(probas)
        probas_test = probas_array[:, :, 1].T
        test = pd.DataFrame(probas_test, columns=DRUG)
        cols = test.apply(lambda s: s.abs().nlargest(7).index.tolist(), axis=1)
        drug_recommend = pd.DataFrame(cols, columns=['DrugsRecommend'])

        atc_drug_name = pd.read_csv('DRUG_ATC_FINAL.csv')

        drug_to_use = pd.DataFrame(drug_recommend['DrugsRecommend'][0], columns=['Mã ATC'])
        drug_to_use_final = drug_to_use.merge(atc_drug_name, how='left', on='Mã ATC')

        st._legacy_dataframe(drug_to_use_final)
        st.subheader("_Thuốc chỉ mang tính chất tham khảo_")

if page == 'Các khuyến cáo':  # PAGE 3
    # ==========Get info============
    # header
    st.markdown("""
    <style>
    .big-font {
        font-size:60px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Các khuyến cáo</p>', unsafe_allow_html=True)
    st.success('Trang đang được hoàn thiện. Tính năng sẽ sớm ra mắt.')



    #PREDICTING
    #y_pred = load_clf.predict(df_step1)
    #predictions = load_clf.predict_proba(df_step1)




