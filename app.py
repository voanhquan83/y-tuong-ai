import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Smart Hospital Initiatives",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Styles - Premium Light Medical Theme */
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;600&family=Inter:wght@300;400;600&display=swap');

    body {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
        background-color: #f8f9fa;
    }
    
    .stApp {
        background-color: #f4f7f6;
        background-image: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* Primary Colors */
    :root {
        --primary-color: #00B4DB; /* Bright Teal */
        --primary-gradient: linear-gradient(to right, #00B4DB, #0083B0);
        --secondary-color: #0083B0; /* Deep Blue */
        --text-color: #2c3e50;
        --card-bg: #ffffff;
        --card-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }

    /* Headers */
    h1, h2, h3, .hero-title, div[data-testid="stMarkdownContainer"] p {
        font-family: 'Exo 2', sans-serif !important;
        color: #1a2a3a !important;
        text-shadow: none !important;
    }

    /* Hero Section */
    .hero-container {
        padding: 4rem 2rem;
        background: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.03);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #546e7a !important;
        font-weight: 400;
    }

    /* Card Layout - Floating Clean Style */
    .card {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid var(--primary-color);
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
        height: 100%;
        color: var(--text-color);
        border: 1px solid #f1f3f4;
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0, 180, 219, 0.15);
        border-color: var(--primary-color);
    }
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #2c3e50;
    }
    .card-desc {
        font-size: 1rem;
        color: #546e7a;
        line-height: 1.6;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        color: #78909c;
        border-radius: 6px;
        border: none;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 180, 219, 0.1) !important;
        color: var(--secondary-color) !important;
        border-bottom: 3px solid var(--secondary-color);
    }
    
    /* Global Text Fix */
    .stMarkdown, .stText, p, li {
        color: #37474f !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }
    
</style>
""", unsafe_allow_html=True)

# Data: 10 AI Solutions
solutions = [
    {
        "id": 1,
        "title": "Dự báo nhu cầu Thuốc & Vật tư tiêu hao",
        "category": "Vận hành",
        "icon": "📦",
        "summary": "Tối ưu hóa tồn kho và giảm lãng phí thông qua dự báo chuỗi cung ứng bằng AI.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Lãng phí do hết hạn:** Các bệnh viện thường đối mặt với việc hủy bỏ lượng lớn thuốc và vật tư y tế quá hạn sử dụng, gây thiệt hại tài chính đáng kể.
        - **Gián đoạn điều trị (Stockouts):** Việc thiếu hụt thuốc đột ngột (đặc biệt là thuốc cấp cứu hoặc đặc trị) có thể đe dọa tính mạng người bệnh và làm gián đoạn quy trình khám chữa bệnh.
        - **Vốn tồn kho:** Duy trì lượng tồn kho quá lớn gây ứ đọng vốn, trong khi tồn kho quá nhỏ lại rủi ro thiếu hụt.
        - **Quản lý thủ công:** Việc dự trù dựa trên kinh nghiệm hoặc Excel thường thiếu chính xác do không tính đến các yếu tố mùa vụ, dịch bệnh bùng phát hay xu hướng thay đổi.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Dữ liệu đầu vào:** Lịch sử nhập/xuất kho trong 3-5 năm, dữ liệu sử dụng thuốc theo bệnh lý, yếu tố mùa vụ (dịch cúm, sốt xuất huyết), và thông tin hạn sử dụng.
        - **Mô hình thuật toán:**
            - Sử dụng **ARIMA (AutoRegressive Integrated Moving Average)** cho các chuỗi dữ liệu có tính quy luật rõ ràng.
            - Ứng dụng **LSTM (Long Short-Term Memory)** - một mạng nơ-ron hồi quy (RNN) để học các mẫu phức tạp và phi tuyến tính từ chuỗi thời gian dài.
        - **Quy trình:**
            1. Xây dựng Data Warehouse, Snowflake và Airflow để tự động thu thập dữ liệu tồn kho.
            2. AI phân tích xu hướng và đưa ra dự báo nhu cầu cho 1-3 tháng tới.
            3. Hệ thống cảnh báo tự động khi lượng tồn kho chạm ngưỡng an toàn (Reorder Point) hoặc khi có thuốc sắp hết hạn (FEFO - First Expired First Out).
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Giảm lãng phí:** Giảm 20-30% lượng thuốc/vật tư phải hủy do hết hạn.
        - **Đảm bảo cung ứng:** Giảm 90% các sự cố thiếu thuốc đột xuất (stockouts).
        - **Tối ưu dòng tiền:** Giảm lượng vốn ứ đọng trong kho khoảng 15%.
        - **Hiệu quả vận hành:** Tiết kiệm hàng trăm giờ làm việc của dược sĩ trong việc lập dự trù thủ công.
        """
    },
    {
        "id": 2,
        "title": "Phân loại bệnh dựa trên xét nghiệm",
        "category": "Lâm sàng",
        "icon": "🩸",
        "summary": "Sàng lọc sớm các rủi ro bệnh lý như Tiểu đường, Suy thận từ kết quả xét nghiệm.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Sàng lọc muộn:** Nhiều bệnh mãn tính (Tiểu đường, CKD) diễn tiến âm thầm, thường chỉ được phát hiện khi đã có biến chứng.
        - **Quá tải bác sĩ:** Bác sĩ phải đọc hàng trăm kết quả xét nghiệm mỗi ngày, dễ dẫn đến bỏ sót các dấu hiệu cảnh báo sớm tinh vi.
        - **Cá nhân hóa:** Các chỉ số tham chiếu chuẩn thường chung chung, chưa tối ưu cho từng cá thể với tiền sử khác nhau.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Dữ liệu đầu vào:** Các chỉ số sinh hóa máu (Glucose, HbA1c, Creatinine, Urea...), tổng phân tích tế bào máu, nước tiểu kèm theo thông tin nhân khẩu học (tuổi, giới tính).
        - **Mô hình thuật toán:**
            - Sử dụng **Random Forest** hoặc **XGBoost** (Gradient Boosting) để phân loại rủi ro. Các mô hình này hoạt động tốt với dữ liệu dạng bảng (tabular data) và có khả năng giải thích (feature importance) để chỉ ra chỉ số nào gây ra nguy cơ cao.
        - **Quy trình:**
            1. Kết quả từ hệ thống LIS (Laboratory Information System) được đẩy vào mô hình AI.
            2. AI đánh giá xác suất mắc bệnh (ví dụ: 85% nguy cơ Tiểu đường tuýp 2).
            3. Cảnh báo "Flag" màu đỏ/vàng trên màn hình bác sĩ với gợi ý làm thêm xét nghiệm chuyên sâu.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Phát hiện sớm:** Tăng tỷ lệ phát hiện sớm bệnh mãn tính lên 25%.
        - **Hỗ trợ ra quyết định:** Giảm sai sót bỏ sót bệnh (missed diagnosis).
        - **Y tế dự phòng:** Chuyển đổi mô hình từ chữa bệnh sang phòng bệnh, giảm chi phí điều trị biến chứng lâu dài.
        """
    },
    {
        "id": 3,
        "title": "Dự báo tiên lượng NB (Prognosis)",
        "category": "Lâm sàng",
        "icon": "vitals", 
        "icon_char": "📈", # Fallback
        "summary": "Tích hợp đa phương thức (Lab, Hình ảnh, Sinh hiệu) để dự đoán tử vong hoặc nhu cầu ICU.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Đánh giá rủi ro:** Khó khăn trong việc xác định nhanh bệnh nhân nào đang chuyển nặng (deteriorating) trong bối cảnh cấp cứu quá tải.
        - **Phân bổ nguồn lực:** Cần quyết định chính xác ai cần giường ICU, ai cần thở máy sớm để tối ưu nguồn lực khan hiếm.
        - **Hỗ trợ gia đình:** Cung cấp thông tin tiên lượng dựa trên dữ liệu để bác sĩ tư vấn giải thích cho người nhà bệnh nhân.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Dữ liệu đầu vào (Multi-modal):** 
            - Dữ liệu có cấu trúc: Lab, Sinh hiệu (Mạch, SPO2, HA).
            - Dữ liệu phi cấu trúc: Kết quả Chẩn đoán hình ảnh (X-quang, CT).
        - **Mô hình thuật toán:** Xây dựng mô hình **Fusion Neural Network** kết hợp CNN (cho hình ảnh) và MLP (cho dữ liệu bảng) để đưa ra điểm số rủi ro tổng hợp (Risk Score).
        - **Quy trình:**
            1. Hệ thống theo dõi thời gian thực (Real-time monitoring) các chỉ số sinh tồn.
            2. Cập nhật liên tục điểm số tiên lượng (ví dụ: SOFA score tự động, APACHE II AI-enhanced).
            3. Cảnh báo bác sĩ nếu xác suất tử vong hoặc biến chứng tăng đột biến trong 6-12 giờ tới.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Cứu sống bệnh nhân:** Giảm tỷ lệ tử vong bệnh viện nhờ can thiệp sớm (Early Intervention).
        - **Tối ưu ICU:** Giảm 15% thời gian nằm ICU không cần thiết nhờ phân loại đầu vào chính xác.
        - **Cải thiện quy trình:** Chuẩn hóa quy trình đánh giá mức độ nặng của người bệnh.
        """
    },
    {
        "id": 4,
        "title": "Dự đoán khả năng tái nhập viện",
        "category": "Vận hành",
        "icon": "🏥",
        "summary": "Phân tích bệnh sử để giảm thiểu tỷ lệ tái nhập viện và cải thiện kế hoạch xuất viện.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Chi phí & Phạt:** Tái nhập viện (trong vòng 30 ngày) gây tốn kém chi phí BHYT và gia đình, đồng thời ảnh hưởng đến chỉ số chất lượng bệnh viện.
        - **Chăm sóc chưa liền mạch:** Bệnh nhân thường xuất viện khi chưa ổn định hoàn toàn hoặc thiếu hướng dẫn chăm sóc tại nhà phù hợp.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Dữ liệu đầu vào:** Hồ sơ bệnh án điện tử (EMR), mã bệnh ICD-10, thuốc đã kê, và tóm tắt xuất viện (Discharge Summary).
        - **Mô hình thuật toán:**
            - Sử dụng **NLP (BERT hoặc ClinicalBERT)** để đọc hiểu văn bản tóm tắt quá trình điều trị.
            - Kết hợp với mô hình Logistic Regression trên các biến số lâm sàng để dự báo xác suất quay lại viện.
        - **Quy trình:**
            1. Tại thời điểm bác sĩ làm thủ tục xuất viện, AI chấm điểm nguy cơ tái nhập viện.
            2. Nếu nguy cơ cao > 70%, hệ thống gợi ý giữ lại theo dõi thêm hoặc thiết lập kế hoạch chăm sóc tại nhà (Homecare) đặc biệt.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Chất lượng điều trị:** Giảm tỷ lệ tái nhập viện trong 30 ngày xuống 10-15%.
        - **Sự hài lòng:** Tăng trải nghiệm người bệnh nhờ sự quan tâm và kế hoạch hậu mãi chu đáo.
        - **Uy tín:** Nâng cao xếp hạng chất lượng bệnh viện.
        """
    },
    {
        "id": 5,
        "title": "Phát hiện NB té ngã qua Camera",
        "category": "Chăm sóc",
        "icon": "📷",
        "summary": "Giám sát an toàn người bệnh bằng Computer Vision để cảnh báo té ngã tức thời.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **An toàn người bệnh:** Té ngã là sự cố y khoa nghiêm trọng hàng đầu tại bệnh viện, đặc biệt với người cao tuổi, gây chấn thương sọ não, gãy xương.
        - **Giám sát 24/7:** Điều dưỡng không thể túc trực cạnh giường bệnh 24/24. Camera giám sát thông thường cần người xem liên tục.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Công nghệ:** Computer Vision (Thị giác máy tính).
        - **Mô hình thuật toán:** 
            - Sử dụng **Pose Estimation (MediaPipe, OpenPose)** để xác định khung xương cơ thể.
            - Hoặc **YOLO (You Only Look Once)** để phát hiện hành động ngã (nằm bất động trên sàn, tư thế bất thường).
        - **Bảo mật:** Xử lý tại biên (Edge AI) hoặc làm mờ mặt để đảm bảo quyền riêng tư.
        - **Quy trình:**
            1. Camera trong phòng bệnh/hành lang phân tích luồng video thời gian thực.
            2. Nhận diện hành động "Ngã".
            3. Gửi cảnh báo khẩn cấp (âm thanh/tin nhắn) đến trạm điều dưỡng ngay lập tức.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Phản ứng nhanh:** Giảm thời gian phát hiện té ngã từ hàng giờ xuống còn vài giây.
        - **Giảm chấn thương:** Cấp cứu kịp thời giúp giảm mức độ nghiêm trọng của chấn thương.
        - **An tâm:** Tăng sự an tâm cho người nhà và giảm áp lực trực gác cho điều dưỡng.
        """
    },
    {
        "id": 6,
        "title": "Trợ lý sức khoẻ tâm thần (AI Chatbot)",
        "category": "Chăm sóc",
        "icon": "🧠",
        "summary": "Hỗ trợ 24/7 với Chatbot NLP sử dụng liệu pháp hành vi nhận thức (CBT).",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Nhu cầu cao:** Rối loạn lo âu, trầm cảm gia tăng nhưng nhân lực bác sĩ tâm lý thiếu hụt.
        - **Rào cản tâm lý:** Bệnh nhân thường ngại chia sẻ trực tiếp với bác sĩ do sợ kỳ thị.
        - **Khoảng trống dịch vụ:** Thiếu sự hỗ trợ ngoài giờ hành chính.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Công nghệ:** Generative AI / NLP.
        - **Mô hình thuật toán:** Sử dụng LLM (Large Language Model) được tinh chỉnh (fine-tuned) trên các nguyên tắc **CBT (Cognitive Behavioral Therapy)**.
        - **Tính năng:**
            - Trò chuyện thấu cảm, lắng nghe tích cực.
            - Phân tích cảm xúc (Sentiment Analysis) để phát hiện xu hướng tiêu cực hoặc ý định tự hại.
        - **Quy trình:**
            1. Bệnh nhân chat qua ứng dụng mobile.
            2. Bot đánh giá tâm trạng và đưa ra bài tập thư giãn/tư vấn tâm lý nhẹ nhàng.
            3. Nếu phát hiện nguy cơ cao (Red flag), bot tự động chuyển cuộc gọi kết nối với chuyên gia con người.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Tiếp cận:** Mở rộng khả năng tiếp cận dịch vụ sức khỏe tâm thần 24/7.
        - **Sàng lọc:** Giúp phân loại mức độ nặng nhẹ để bác sĩ ưu tiên điều trị ca nặng.
        - **Hỗ trợ điều trị:** Giúp bệnh nhân tuân thủ phác đồ và có người đồng hành cảm xúc.
        """
    },
    {
        "id": 7,
        "title": "Trợ lý chăm sóc người bệnh mãn tính",
        "category": "Chăm sóc",
        "icon": "⌚",
        "summary": "Tích hợp IoT và AI để cá nhân hóa lối sống cho bệnh nhân Tiểu đường/Huyết áp.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Quản lý tại nhà:** Bệnh mãn tính cần quản lý liên tục tại nhà, không chỉ ở bệnh viện.
        - **Tuân thủ kém:** Bệnh nhân thường quên uống thuốc, ăn uống không đúng chế độ dẫn đến bệnh tái phát.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Dữ liệu:** Kết nối với thiết bị IoT (Đồng hồ đo nhịp tim, máy đo đường huyết Bluetooth).
        - **Mô hình thuật toán:** Recommendation System (Hệ gợi ý) dựa trên dữ liệu sinh học cá nhân.
        - **Quy trình:**
            1. Thu thập dữ liệu vận động, chỉ số đường huyết hàng ngày.
            2. AI phân tích và gửi "Nudges" (Lời nhắc) cá nhân hóa: "Hôm nay đường huyết hơi cao, bạn nên đi bộ thêm 10 phút và giảm tinh bột bữa tối."
            3. Báo cáo định kỳ gửi về bác sĩ điều trị.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Kiểm soát bệnh:** Ổn định chỉ số đường huyết/huyết áp tốt hơn 30% so với tự quản lý.
        - **Gắn kết:** Tăng sự tương tác giữa bệnh viện và bệnh nhân sau xuất viện.
        - **Dữ liệu thực:** Bác sĩ có dữ liệu thực tế (Real-world evidence) để điều chỉnh thuốc chính xác.
        """
    },
    {
        "id": 8,
        "title": "Tra cứu hồ sơ bệnh án (RAG Assistant)",
        "category": "Vận hành",
        "icon": "🔍",
        "summary": "Hỏi đáp ngôn ngữ tự nhiên về lịch sử bệnh nhân giúp bác sĩ tiết kiệm thời gian.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Quá tải thông tin:** Hồ sơ bệnh án điện tử của bệnh nhân lâu năm có thể dài hàng trăm trang.
        - **Mất thời gian:** Bác sĩ mất rất nhiều thời gian để "đào bới" (click chuột, cuộn trang) tìm thông tin tiền sử dị ứng, tiền sử phẫu thuật cũ trong tình huống cấp cứu.
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Công nghệ:** RAG (Retrieval-Augmented Generation) kết hợp LLM.
        - **Quy trình:**
            1. Vector hóa (Embedding) toàn bộ dữ liệu lịch sử khám, phiếu phẫu thuật, kết quả xét nghiệm cũ của bệnh nhân vào Vector Database.
            2. Bác sĩ hỏi: "Bệnh nhân này có tiền sử dị ứng kháng sinh không?" hoặc "Lần mổ ruột thừa gần nhất là khi nào?"
            3. Hệ thống truy xuất đoạn văn bản liên quan và dùng LLM tóm tắt câu trả lời ngắn gọn kèm trích dẫn nguồn (Evidence).
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Tiết kiệm thời gian:** Giảm 70% thời gian tra cứu hồ sơ.
        - **Chính xác:** Cung cấp thông tin nhanh và chính xác ngay tại điểm chăm sóc (Point-of-care).
        - **Trải nghiệm:** Giảm thao tác máy tính, giúp bác sĩ tập trung vào thăm khám bệnh nhân.
        """
    },
    {
        "id": 9,
        "title": "Phân tích lượng máu mất trong phẫu thuật",
        "category": "Lâm sàng",
        "icon": "🩸",
        "summary": "Sử dụng Computer Vision để tính toán chính xác lượng máu mất qua gạc phẫu thuật.",
        "necessity": """
        **Tại sao cần giải pháp này?**
        - **Ước lượng chủ quan:** Hiện tại bác sĩ/điều dưỡng thường ước lượng máu mất bằng mắt thường ("nhìn gạc đoán máu"), độ chính xác thấp.
        - **Quyết định truyền máu:** Ước lượng sai dẫn đến truyền máu muộn (nguy hiểm) hoặc truyền máu thừa (lãng phí và rủi ro phản ứng).
        """,
        "implementation": """
        **Cách thức hoạt động của AI:**
        - **Công nghệ:** Computer Vision & Colorimetric Analysis.
        - **Thiết bị:** iPad hoặc Camera chuyên dụng trong phòng mổ.
        - **Quy trình:**
            1. Chụp ảnh gạc thấm máu sau khi sử dụng.
            2. AI phân tích độ bão hòa màu sắc (Huyết sắc tố Hemoglobin) trên vải gạc.
            3. Tính toán quy đổi ra thể tích máu thực tế đã mất.
            4. Hiển thị tổng lượng máu mất theo thời gian thực trên màn hình phòng mổ.
        """,
        "results": """
        **Kết quả & Lợi ích (ROI):**
        - **Chính xác:** Độ chính xác > 90% so với xét nghiệm máu.
        - **An toàn:** Ra quyết định truyền máu kịp thời, cứu sống bệnh nhân trong các ca đại phẫu.
        - **Tiết kiệm:** Quản lý nguồn máu hiến tặng hiệu quả hơn.
        """
    },
]

# --- POC DEMO FUNCTION ---
def render_poc(solution_id, solution_title):
    st.markdown(f"#### 🔮 Demo: {solution_title}")
    
    if solution_id == 1: # Supply Chain
        st.write("Mô phỏng dự báo nhu cầu thuốc Paracetamol 500mg:")
        chart_data = pd.DataFrame({
            'Thực tế': np.random.randint(50, 100, 30),
            'Dự báo AI': np.random.randint(55, 95, 30)
        })
        st.line_chart(chart_data)
        st.success("✅ AI phát hiện xu hướng tăng nhu cầu vào cuối tháng!")

    elif solution_id == 2: # Disease Classification
        c1, c2 = st.columns(2)
        with c1:
            glucose = st.slider("Chỉ số Glucose (mg/dL)", 70, 200, 145)
            hba1c = st.slider("Chỉ số HbA1c (%)", 4.0, 10.0, 7.2)
        with c2:
            bmi = st.slider("BMI", 15.0, 40.0, 28.5)
        
        if st.button("🔍 Phân tích rủi ro", key="btn_poc_2"):
            risk = (glucose * 0.4 + hba1c * 10 + bmi) / 3
            if risk > 50:
                st.error(f"⚠️ Nguy cơ cao Tiểu đường (Score: {risk:.1f}). Đề xuất khám chuyên khoa.")
            else:
                st.success(f"🟢 Nguy cơ thấp (Score: {risk:.1f}).")

    elif solution_id == 3: # Prognosis
        st.write("Nhập chỉ số sinh hiệu bệnh nhân ICU:")
        c1, c2, c3 = st.columns(3)
        spo2 = c1.number_input("SpO2 (%)", 50, 100, 92)
        hr = c2.number_input("Nhịp tim (BPM)", 40, 200, 110)
        bp = c3.selectbox("Huyết áp", ["Bình thường", "Thấp", "Cao"], index=2)
        
        if spo2 < 95 and hr > 100:
            st.warning("⚠️ Cảnh báo: Nguy cơ suy hô hấp trong 4h tới là 85%!")
            st.progress(85)
        else:
            st.info("Ổn định. Tiếp tục theo dõi.")

    elif solution_id == 4: # Readmission
        st.caption("Phân tích hồ sơ bệnh án tự động...")
        st.text_area("Tóm tắt xuất viện (Simulation Log)", "Bệnh nhân nam, 65 tuổi, tiền sử suy tim độ 3. Điều trị ổn định...", height=100, disabled=True)
        st.metric("Xác suất tái nhập viện (30 ngày)", "72%", "+15%")
        st.info("💡 Đề xuất: Kích hoạt gói Homecare theo dõi huyết áp tại nhà.")

    elif solution_id == 5: # Fall Detection
        st.info("📷 Live Feed giả lập từ Camera phòng 304")
        # Creating a fake 'video feed' placeholder
        st.markdown(
            """
            <div style="background-color: #000; height: 200px; color: white; display: flex; align-items: center; justify-content: center; border: 2px solid red;">
                🎥 [AI YOLOv8] Detecting Person... <br>
                ⚠️ ACTION: FALL DETECTED (Confidence: 99.8%)
            </div>
            """, unsafe_allow_html=True
        )
        st.button("🔴 GỬI BÁO ĐỘNG NGAY", type="primary")

    elif solution_id == 6: # Mental Health Chatbot
        user_input = st.text_input("Trò chuyện với AI:", placeholder="Tôi cảm thấy lo lắng quá...")
        if user_input:
            st.markdown("""
            **🤖 Chatbot:** "Tôi hiểu bạn đang cảm thấy không ổn. Bạn có thể chia sẻ cụ thể hơn điều gì khiến bạn lo lắng không? Hít thở sâu nhé, tôi ở đây để lắng nghe."
            """)

    elif solution_id == 7: # Chronic Care
        st.write("Dữ liệu từ Smartwatch & IoT:")
        st.metric("Bước chân hôm nay", "2,341", "-500 so với mục tiêu")
        st.warning("🔔 Nhắc nhở: Bạn chưa đạt mục tiêu vận động. Hãy đi bộ nhẹ 15 phút sau bữa tối nhé!")

    elif solution_id == 8: # Drug Interaction
        d1 = st.selectbox("Thuốc 1", ["Aspirin", "Paracetamol", "Insulin"], key="d1")
        d2 = st.selectbox("Thuốc 2", ["Warfarin", "Ibuprofen", "Vitamin C"], key="d2")
        
        if d1 == "Aspirin" and d2 == "Warfarin":
            st.error("⛔ TƯƠNG TÁC NGHIÊM TRỌNG: Nguy cơ chảy máu cao!")
        elif d1 == "Aspirin" and d2 == "Ibuprofen":
            st.warning("⚠️ Tương tác trung bình: Giảm tác dụng bảo vệ tim mạch.")
        else:
            st.success("✅ Không tìm thấy tương tác đáng kể.")

    elif solution_id == 9: # RAG Search
        q = st.text_input("Tra cứu hồ sơ:", "Bệnh nhân có tiền sử dị ứng Penicillin không?")
        if q:
            with st.spinner("Đang đọc hồ sơ (200 trang)..."):
                st.write("**🤖 Kết quả:** Có. Ghi nhận dị ứng Penicillin năm 2018 (Biểu hiện: Nổi mề đay).")
                st.caption("Nguồn: Phiếu khám ngày 12/05/2018 - BS. Nguyễn Văn A.")

    elif solution_id == 10: # Blood Loss
        st.write("📷 Phân tích hình ảnh gạc phẫu thuật:")
        c1, c2 = st.columns(2)
        c1.markdown('<div style="width:100px; height:100px; background-color: #8b0000; border-radius: 5px;"></div> (Ảnh Gạc)', unsafe_allow_html=True)
        with c2:
            st.metric("Lượng máu ước tính", "150 ml", "Chính xác cao")
            st.progress(30, text="Ngưỡng truyền máu: An toàn")

# --- UI LOGIC ---

# Setup Sidebar
with st.sidebar:
    st.title("🏥 Smart Hospital AI")
    st.markdown("---")
    
    st.write("Dùng bộ lọc bên dưới để tìm giải pháp phù hợp:")
    
    # Category Filter
    st.subheader("Lĩnh vực")
    all_categories = sorted(list(set([s['category'] for s in solutions])))
    selected_categories = st.multiselect(
        "Chọn lĩnh vực:",
        options=all_categories,
        default=all_categories
    )
    
    st.markdown("---")
    st.markdown("""
    ### Mục lục nhanh
    - [Giới thiệu](#hero)
    - [Danh sách giải pháp](#solutions)
    - [Liên hệ](#contact)
    """)
    st.markdown("---")
    st.caption("© Tổ KTPM")

# Create a mapping for filter
filtered_solutions = [s for s in solutions if s['category'] in selected_categories]

# --- MAIN LANDING PAGE ---

# 1. HERO SECTION
st.markdown('<a id="hero"></a>', unsafe_allow_html=True)
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Sáng kiến Bệnh viện Thông minh</div>
        <div class="hero-subtitle">10 Giải pháp AI đột phá chuyển đổi số Y tế</div>
        <div style="margin-top: 20px; font-size: 1rem;">
             Khám phá các ứng dụng Trí tuệ nhân tạo giúp nâng cao chất lượng điều trị và tối ưu vận hành
        </div>
    </div>
""", unsafe_allow_html=True)

# 2. SOLUTIONS LIST SECTION
st.markdown('<a id="solutions"></a>', unsafe_allow_html=True)
st.subheader(f"📂 Danh sách giải pháp ({len(filtered_solutions)})")
st.markdown("---")

if not filtered_solutions:
    st.warning("Không tìm thấy giải pháp nào phù hợp với bộ lọc hiện tại.")

for idx, sol in enumerate(filtered_solutions):
    # Create a container for each solution
    with st.container():
        # Header for the solution
        col_icon, col_content = st.columns([1, 15])
        
        with col_icon:
            st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{sol.get('icon_char', sol['icon'])}</div>", unsafe_allow_html=True)
            
        with col_content:
            st.markdown(f"### {sol['title']}")
            st.caption(f"**Lĩnh vực:** {sol['category'].upper()}")
            st.markdown(f"_{sol['summary']}_")
        
        # Detailed Tabs inline
        st.markdown("")
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Tính cần thiết", "⚙️ Cách thực hiện", "🏆 Kết quả đạt được", "🔮 Demo"])
        
        with tab1:
            st.markdown(sol['necessity'])
            
        with tab2:
            st.info("Kiến trúc kỹ thuật & Giải thuật")
            st.markdown(sol['implementation'])
            
        with tab3:
            st.success("ROI & Tác động lâm sàng")
            st.markdown(sol['results'])
            
        with tab4:
            render_poc(sol['id'], sol['title'])
            
    # Add a visual separator between items
    st.markdown("""<hr style="border-top: 3px solid #f0f2f6; margin: 30px 0;">""", unsafe_allow_html=True)

# 3. CONTACT / FOOTER SECTION
st.markdown('<a id="contact"></a>', unsafe_allow_html=True)
st.markdown("""
    <div style="background-color: white; color: #2c3e50; padding: 3rem; border-radius: 12px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
        <h2 style="color: #0083B0 !important;">🤝 Bạn đã sẵn sàng chuyển đổi số?</h2>
        <p style="color: #546e7a;">Kết nối với chúng tôi để được tư vấn lộ trình triển khai.</p>
        <div style="margin-top: 20px;">
            <p style="font-size: 1.1rem; font-weight: bold; color: #37474f;">Phòng Công nghệ thông tin<br>Bệnh viện Đại học Y Dược Thành phố Hồ chí Minh</p>
            <p style="color: #0083B0; font-weight: bold;">Hotline: 1900 2827</p>
        </div>
        <hr style="border-color: #f0f0f0; margin: 20px 0;">
        <p style="font-size: 0.8rem; opacity: 0.7; color: #90a4ae;">© 2024 AI Solutions. Powered by Streamlit.</p>
    </div>
""", unsafe_allow_html=True)
