from configparser import ConfigParser

# This file is for parsing configs generated from config_generator.py.
# No need to run this file.

class Config(ConfigParser):
    def __init__(self, config_file):
        super(Config, self).__init__()
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)



# from configparser import ConfigParser
#
# 这行代码从 configparser 模块导入 ConfigParser 类，该类用于解析配置文件（如 .ini 文件）。
# class Config(ConfigParser):
#
# 定义了一个名为 Config 的类，它继承自 ConfigParser 类。
# def __init__(self, config_file):
#
# 这是 Config 类的构造函数，它接受一个参数 config_file，这个参数是配置文件的路径。
# raw_config = ConfigParser()
#
# 创建一个 ConfigParser 对象实例 raw_config，用于读取和解析配置文件。
# raw_config.read(config_file)
#
# 使用 raw_config 对象的 read 方法读取配置文件。
# self.cast_values(raw_config)
#
# 调用 cast_values 方法，将 raw_config 中的值转换为适当的数据类型，并存储在 Config 类的实例中。
# def cast_values(self, raw_config):
#
# 定义了一个名为 cast_values 的方法，它接受一个参数 raw_config。
# for section in raw_config.sections():
#
# 遍历 raw_config 中的所有章节（section）。
# for key, value in raw_config.items(section):
#
# 对于每个章节，遍历其所有的键值对。
# val = None
#
# 初始化一个变量 val，用于存储转换后的值。
# if type(value) is str and value.startswith("[") and value.endswith("]"):
#
# 检查 value 是否是一个字符串，并且以 [ 开头，以 ] 结尾。这通常表示值是一个列表或元组。
# val = eval(value)
#
# 使用 eval 函数将字符串形式的列表或元组转换为实际的 Python 对象，并赋值给 val。
# setattr(self, key, val)
#
# 使用 setattr 函数将转换后的值 val 赋给 self（当前 Config 类的实例）的属性 key。
# continue
#
# 跳过当前循环的剩余部分，继续下一次迭代。
# for attr in ["getint", "getfloat", "getboolean"]:
#
# 遍历一个包含字符串 "getint"、"getfloat" 和 "getboolean" 的列表，这些是 ConfigParser 类提供的方法，用于获取整型、浮点型和布尔型值。
# try:
#
# 开始一个 try 代码块，用于捕获可能发生的异常。
# val = getattr(raw_config[section], attr)(key)
#
# 尝试使用 getattr 函数获取 raw_config 对象的 attr 方法，并传入 key，以获取转换后的值。
# break
#
# 如果转换成功，则跳出当前的 for 循环，继续下一次章节或键值对的迭代。
# except:
#
# 如果发生异常（例如，值不能转换为期望的类型），则执行 except 代码块。
# val = value
#
# 将原始的 value 赋给 val，这意味着如果转换失败，原始值将被保留。
# setattr(self, key, val)
#
# 最后，无论转换是否成功，都将 val 赋给 self 的属性 key。
# 这段代码的主要作用是读取配置文件，并将其中的值转换为适当的 Python 数据类型，以便在程序中使用。通过继承 ConfigParser 类，它利用了该类提供的功能来处理配置文件中的各种数据类型