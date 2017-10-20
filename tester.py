import requests
import sys


def main(path):
  with open(path, 'rb') as f:
    data = f.read()

  headers = {'rgb_im_len': str(len(data))}
  while True:
    r = requests.post('http://0.0.0.0:5000/api/detect', headers=headers, data=data)
    print(r.text)

if __name__ == '__main__':
  main(sys.argv[1])
