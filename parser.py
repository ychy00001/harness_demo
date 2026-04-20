import re
from typing import List

class EmailParser:
    def __init__(self):
        # 正则表达式说明:
        # [a-zA-Z0-9_-]+ : 匹配一个或多个字母、数字 (用户名部分)
        # @              : 匹配 @ 符号
        # [a-zA-Z0-9_-]+ : 匹配一个或多个字母、数字 (域名部分)
        # \.com          : 匹配 .com 后缀 (注意点号需要转义)
        self.pattern = re.compile(r'[a-zA-Z0-9]+@[a-zA-Z0-9]+\.com')

    def extract_email(self, text: str) -> List[str]:
        """
        从输入的文本中提取符合 字母数字@字母数字.com 格式的邮箱地址。
        
        参数:
            text (str): 用户输入的文本字符串
            
        返回:
            List[str]: 包含所有匹配到的邮箱地址的列表
        """
        if not text:
            return []
            
        # findall 返回所有匹配的字符串列表
        return self.pattern.findall(text)

# --- 测试代码 ---
if __name__ == "__main__":
    parser = EmailParser()
    
    # 模拟用户输入的一堆文本
    sample_text = """
    你好，请联系我。
    我的主邮箱是 user123@example.com。
    备用邮箱是 admin_123@company.com。
    无效的邮箱格式：test@abc.org
    另一个有效的：zhang@163.com
    """
    
    emails = parser.extract_email(sample_text)
    
    print(f"原始文本:\n{sample_text}")
    print("-" * 30)
    print(f"提取到的邮箱列表: {emails}")
