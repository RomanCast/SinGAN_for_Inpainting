import numpy as np
import cv2
import matplotlib.pyplot as plt

def contour_holes(im, p_size):
  p = p_size//2
  kernel = np.ones((50,50),np.uint8)
  dilate = cv2.dilate(im,kernel,iterations = 10)
  ret,thresh = cv2.threshold(im,127,255,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  boxx,boxy,boxw,boxh = 0,0,0,0
  for i in range(0, len(contours)):
      if (i % 2 == 0):
        cnt = contours[i]
        boxx,boxy,boxw,boxh = cv2.boundingRect(cnt)
  return [boxx,boxy,boxw,boxh]

def is_hole(mask, i, j):
  return mask[i,j]==255

def in_box(x, y, w, h, i, j):
  return (j>=x and j<x+w and i>=y and i<y+h)

def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :]
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist

def reconstruction(f, A, B, mask, mask_orig, p_size):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    p=p_size//2
    temp = B.copy()
    temp_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    temp_padding[p:A_h+p, p:A_w+p, :] = temp
    mask_new = mask.copy()
    for i in range(A_h):
        for j in range(A_w):
          if not is_hole(mask,i,j):
            nn=f[i,j]
            i_min = max(i-p,0)
            i_max = min(i+p+1,A_h-1)
            j_min = max(j-p,0)
            j_max = min(j+p+1,A_w-1)
            temp_padding[i:i+p_size,j:j+p_size,:] = B[nn[0]-p:nn[0]+p+1,nn[1]-p:nn[1]+p+1,:]
            mask_new[i_min:i_max,j_min:j_max] = 0

    for i in range(A_h):
      for j in range(A_w):
        if is_hole(mask_orig,i,j):
          temp[i,j,:] = temp_padding[i+p,j+p,:]
    return temp,mask_new

def initialization(A, B, mask, p_size):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    p = p_size // 2
    boxx,boxy,boxw,boxh = contour_holes(mask,p_size)
    B_h_max = min(boxy+boxh+p_size - 1,A_h-p-1)
    B_w_max = min(boxx+boxw+p_size - 1,A_w-p-1)
    B_h_min = max(boxy-p_size,p)
    B_w_min = max(boxx-p_size,p)
    A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    A_padding[p:A_h+p, p:A_w+p, :] = A
    f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])
    for i in range(A_h):
        if i % 100 ==0:
          print(i)
        for j in range(A_w):
          a = np.array([i, j])
          valid = False
          while (not valid):
            by = np.random.randint(B_h_min,B_h_max)
            bx = np.random.randint(B_w_min,B_w_max)
            if not in_box(boxx,boxy,boxw,boxh,max(by-p,0),max(bx-p,0)) and not in_box(boxx,boxy,boxw,boxh,min(by+p,A_h-1),min(bx+p,A_w-1)):
              valid = True
          b = np.array([by,bx])
          f[i, j] = b
          dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding

def propagation(f, a, dist, A_padding, B, box, p_size, is_odd):
    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    p = p_size//2
    x = a[0]
    y = a[1]
    cand = f[x,y]
    boxx,boxy,boxw,boxh = box[0],box[1],box[2],box[3]
    if is_odd:
        d_left = dist[max(x-1, 0), y]
        d_up = dist[x, max(y-1, 0)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1:
            cand = f[max(x - 1, 0), y]
        if idx == 2:
            cand = f[x, max(y - 1, 0)]
        if not in_box(boxx,boxy,boxw,boxh,max(cand[0]-p,0),max(cand[1]-p,0)) and not in_box(boxx,boxy,boxw,boxh,min(cand[0]+p,A_h-1), min(cand[1]+p,A_w-1)):
          f[x,y]=cand
          dist[x,y]= cal_distance(a, cand, A_padding, B, p_size)
    else:
        d_right = dist[min(x + 1, A_h-1), y]
        d_down = dist[x, min(y + 1, A_w-1)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1 :
            cand = f[min(x + 1, A_h-1), y]
        if idx == 2 :
            cand = f[x, min(y + 1, A_w-1)]
        if not in_box(boxx,boxy,boxw,boxh,max(cand[0]-p,0),max(cand[1]-p,0)) and not in_box(boxx,boxy,boxw,boxh,min(cand[0]+p,A_h-1), min(cand[1]+p,A_w-1)):
            f[x, y] = cand
            dist[x, y] = cal_distance(a, cand, A_padding, B, p_size)

def random_search(f, a, dist, A_padding, B, box, p_size, alpha=0.5):
    x = a[0]
    y = a[1]
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    boxx,boxy,boxw,boxh = box[0],box[1],box[2],box[3]
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h-p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        search_min_c = max(b_y - search_w, p)
        search_max_c = min(b_y + search_w, B_w - p)
        random_b_y = np.random.randint(search_min_c, search_max_c)
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y] and not in_box(boxx,boxy,boxw,boxh,max(b[0]-p,0),max(b[1]-p,0)) and not in_box(boxx,boxy,boxw,boxh,min(b[0]+p,B_h-1),min(b[1]+p,B_w-1)):
            dist[x, y] = d
            f[x, y] = b
        i += 1

def NNS(img, mask, p_size, itr):
    A_h = np.size(img, 0)
    A_w = np.size(img, 1)
    ref = A.copy()
    p = p_size//2
    box = contour_holes(mask,p_size)
    boxx,boxy,boxw,boxh = box[0],box[1],box[2],box[3]
    mask_orig=mask.copy()
    f, dist, img_padding = initialization(img, ref, mask,p_size)
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(min(boxy+boxh+p_size - 1,A_h-1), max(boxy-p_size-1,0), -1):
                if i % 100 ==0:
                  print(i)
                for j in range(min(boxx+boxw+p_size - 1,A_w-1), max(boxx-p_size-1,0), -1):
                  #si on est centre sur un pixel
                  #dont le patch en entier n'est pas dans la zone manquante
                  if not is_hole(mask,i,j): #or not is_hole(mask,min(i+1,A_h-1),min(j+1,A_w-1)):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, box,p_size, False)
                    random_search(f, a, dist, img_padding, ref, box,p_size)
        else:
            for i in range(max(boxy-p_size,0),min(boxy+boxh+p_size,A_h-1)):
                if i % 100 ==0:
                  print(i)
                for j in range(max(boxx-p_size,0),min(boxx+boxw+p_size,A_w-1)):
                  if not is_hole(mask,i,j): # or not is_hole(mask,max(i-1,0),max(j-1,0)):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, box,p_size, True)
                    random_search(f, a, dist, img_padding, ref, box,p_size)
        print("iteration: %d"%(itr))
        nns = np.zeros_like(img)
        for i in range(A_h):
          for j in range(A_w):
            nns[i,j,:] = ref[f[i,j][0],f[i,j][1],:]
        img,mask = reconstruction(f,img,ref,mask,mask_orig,p_size)
        img_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
        img_padding[p:A_h+p, p:A_w+p, :] = img
        print('nns')
        plt.imshow(nns)
        plt.show()
        print('reconstruction')
        plt.imshow(img)
        plt.show()

    return f,img
