import logging

from inference import inference


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
    inference('c1ccccc1', 'MRGARGAWDFLCVLLLLLR')
