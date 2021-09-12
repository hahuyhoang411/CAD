#=========IMPORT LIBRARY===========
import streamlit as st
st.set_page_config(layout="wide")

#library
import pandas as pd
import pickle
import numpy as np
from sklearn import metrics

#model
import xgboost as xgb
from xgboost import XGBClassifier

#===SET UP 3 PAGES SELECTION===


#SIDEBAR
new_title = '<p style="font-size: 42px; font-style: Bold">Tên đứa con</p>'
st.sidebar.markdown(new_title, unsafe_allow_html=True)
st.sidebar.markdown('Phần mềm dự đoán tỷ lệ sống và thuốc phù hợp cho bệnh nhân mắc bệnh tim thiếu máu cục bộ')
st.sidebar.title("-------------------------------")
page = st.sidebar.radio("Chuyển tới:", options = ['Giới thiệu','Nhập số liệu và dự đoán','Các khuyến cáo'],key='1')

if page == 'Giới thiệu': #PAGE 1
    #===========NAME============
    st.title('Tên của đứa con')

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
    st.title("File .csv của bệnh nhân")
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
                                ('Da trắng', 'Da vàng', 'Da đen',
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
                bmi = demo2.number_input('Chỉ số BMI:')
                stay = demo2.number_input('Lần nhấp viện thứ:',step=1)

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
                if option_pro == '(51) Cấy máy khử rung tim tái đồng bộ, hệ thống tổng thể [CRT-D]': #1
                    PROCE_51 = 1
                elif option_pro == '(54)Chỉ cấy hoặc thay thế máy phát xung máy khử rung tim tái đồng bộ hóa tim [CRT-D]':#2
                    PROCE_54 = 1
                elif option_pro == '(66)Nong mạch vành qua da [PTCA]':#3
                    PROCE_66 = 1
                elif option_pro == '(3603)Nong động mạch vành ngực hở':#4
                    PROCE_3603 = 1
                elif option_pro == '(3606) Đặt (các) stent động mạch vành không dùng thuốc rửa giải':#5
                    PROCE_3606 = 1
                elif option_pro == '(3607) Đặt (các) stent động mạch vành rửa giải bằng thuốc]':#6
                    PROCE_3607 = 1
                elif option_pro == '(3611)(Động mạch chủ) bắc cầu mạch vành của một động mạch vành':#7
                    PROCE_3611 = 1
                elif option_pro == '(3612) (Động mạch chủ) bắc cầu mạch vành của hai động mạch vành':#8
                    PROCE_3612 = 1
                elif option_pro == '(3613)(Động mạch chủ) bắc cầu mạch vành của ba động mạch vành':#9
                    PROCE_3613 = 1
                elif option_pro == '(3722) Thông tim trái':#10
                    PROCE_3722 = 1
                elif option_pro == '(3614)(Động mạch chủ) bắc cầu mạch vành của bốn hoặc nhiều động mạch vành':#11
                    PROCE_3614 = 1
                elif option_pro == '(3615) Cầu nối động mạch vành-động mạch vú đơn bên trong':#12
                    PROCE_3615 = 1
                elif option_pro == '(3616) Cầu nối động mạch vành - động mạch vú đôi bên trong':#13
                    PROCE_3616 = 1
                elif option_pro == '(3617) Bắc cầu động mạch vành bụng':#14
                    PROCE_3617 = 1
                elif option_pro == '(3721) Thông tim phải':#15
                    PROCE_3721 = 1
                elif option_pro == '(3723) Kết hợp thông tim phải và tim trái':#16
                    PROCE_3723 = 1
                elif option_pro == '(3768) Lắp thiết bị trợ tim ngoài qua da':#17
                    PROCE_3768 = 1
                elif option_pro == '(3778) Lắp đặt hệ thống máy tạo nhịp tim truyền tĩnh mạch tạm thời':#18
                    PROCE_3778 = 1
                elif option_pro == '(3795) Chỉ cấy(các) dây dẫn máy khử rung tim / máy khử rung tim tự động':#19
                    PROCE_3795 = 1
                elif option_pro == '(3796) Chỉ cấy máy phát xung máy khử rung tim / máy khử rung tim tự động':#20
                    PROCE_3796 = 1
                elif option_pro == '(3797) Chỉ thay thế(các) dây dẫn máy khử rung tim / máy khử rung tim tự động':#21
                    PROCE_3797 = 1
                elif option_pro == '(3798) Chỉ thay thế máy phát xung máy khử rung tim / máy khử rung tim tự động':#22
                    PROCE_3798 = 1
                elif option_pro == '(3964) Máy tạo nhịp tim trong phẫu thuật':#23
                    PROCE_3964 = 1
                elif option_pro == '(8855) Chụp động mạch vành bằng một ống thông duy nhất':#24
                    PROCE_8855 = 1
                else:
                    PROCE_8856 = 1

                # ===PART2===
                with st.expander("Chỉ số xét nghiệm"):

                    vit_head, lab_head = st.columns([1, 1])
                    vit_head.header('Chỉ số thông thường')
                    lab_head.header('Chỉ số đặc trưng')

                    vital1, vital2, vital3, lab = st.columns([1, 1, 1, 1])

                    heartrate_min = vital1.number_input('Nhịp tim nhỏ nhất:')
                    sysbp_min = vital1.number_input('Huyết áp nhỏ nhất:')
                    diasbp_min = vital1.number_input('Huyết áp tâm thu nhỏ nhất:')
                    meanbp_min = vital1.number_input('Huyết áp tâm trương nhỏ nhất:')
                    resprate_min = vital1.number_input('Nhịp thở nhỏ nhất:')
                    tempc_min = vital1.number_input('Nhiệt độ nhỏ nhất:')
                    spo2_min = vital1.number_input('SpO2 nhỏ nhất:')
                    glucose_min = vital1.number_input('Glucose nhỏ nhất:')

                    heartrate_max = vital2.number_input('Nhịp tim lớn nhất:')
                    sysbp_max = vital2.number_input('Huyết áp lớn nhất:')
                    diasbp_max = vital2.number_input('Huyết áp tâm thu lớn nhất:')
                    meanbp_max = vital2.number_input('Huyết áp tâm trương lớn nhất:')
                    resprate_max = vital2.number_input('Nhịp thở lớn nhất:')
                    tempc_max = vital2.number_input('Nhiệt độ lớn nhất:')
                    spo2_max = vital2.number_input('SpO2 lớn nhất:')
                    glucose_max = vital2.number_input('Glucose lớn nhất:')

                    heartrate_mean = vital3.number_input('Nhịp tim trung bình:')
                    sysbp_mean = vital3.number_input('Huyết áp trung bình:')
                    diasbp_mean = vital3.number_input('Huyết áp tâm thu trung bình:')
                    meanbp_mean = vital3.number_input('Huyết áp tâm trương trung bình:')
                    resprate_mean = vital3.number_input('Nhịp thở trung bình:')
                    tempc_mean = vital3.number_input('Nhiệt độ trung bình:')
                    spo2_mean = vital3.number_input('SpO2 trung bình:')
                    glucose_mean = vital3.number_input('Glucose trung bình:')

                    # LAB_FEATURE
                    choles = lab.number_input('Tỉ lệ cholesterol:')
                    choles_total = lab.number_input('Tổng lượng cholesterol:')
                    hdl = lab.number_input('HDL:')
                    ldl = lab.number_input('LDL:')
                    tri = lab.number_input('Triglycerids:')
                    tro = lab.number_input('Troponint:')
                    ck_mb = lab.number_input('Ckmb:')
                    ck = lab.number_input('Ck:')

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

                albumin = piv.number_input('Albumin:')
                anion = piv.number_input('Aninongap:')
                bicar = piv.number_input('Bicarbonate:')
                bili = piv.number_input('Bilirubin:')
                crea = piv.number_input('Creatinine:')
                chlo = piv.number_input('Chloride:')
                glu = piv.number_input('Glucose:')
                hema = piv.number_input('Hematocrit:')
                hemo = piv.number_input('Hemoglobin:')

                lac = piv2.number_input('Lactate:')
                Plate = piv2.number_input('Platelet:')
                potassium = piv2.number_input('Kali:')
                ptt = piv2.number_input('aPTT:')
                inr = piv2.number_input('INR')
                pt = piv2.number_input('PT')
                sodium = piv2.number_input('Natri:')
                bun = piv2.number_input('BUN:')
                wbc = piv2.number_input('WBC:')

                #NOT DEFINE YET
                los_hos = 0
                first_hosp_stay = 0
                los_icu = 0
                first_icu_stay = 0
                first_icu_stay_demo = 0
                seq_num = 0
                DIAG_4111, DIAG_4139, DIAG_4142, DIAG_4148, DIAG_41001, DIAG_41011, DIAG_41021, DIAG_41031, DIAG_41041, DIAG_41071, DIAG_41072, DIAG_41081,DIAG_41091, DIAG_41189, DIAG_41400, DIAG_41401, DIAG_41402,DIAG_41412 = 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0,0,0,0,0,0

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
                    'PROCE_8856': PROCE_8856}

            features = pd.DataFrame(data, index=[0])
            return features

        users_input_df = user_input_features()

        if uploaded_file is not None:
            st.write(users_input_df)
        else:
            st.subheader('_Đang chờ người dùng nhập file .csv. Hiện tại phần mềm đang sử dụng dữ liệu nhập tay._')
            st.write(users_input_df)

        # IMPORT DATA
        cad_raw = pd.read_csv('MIMIC3_CAD.csv')
        DRUG = [c_ for c_ in cad_raw if c_.startswith('DRUG')]  # drug for CAD
        drop = DRUG + ['hospital_expire_flag'] + ['hadm_id']
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
            st.title('Dư đoán tỷ lệ sống của bệnh nhân')
            pred, proba = st.columns([1, 1])

            pred.header('_Tình trạng bệnh nhân:_')
            if prediction[0] == 0:
                pred.success('Sống')
            else:
                pred.warning('Chết')
            #pred.write(prediction[0])
            proba.header('_Tỷ lệ sống của bệnh nhân_')

            proba.success(prediction_proba[0][0]) #%



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

