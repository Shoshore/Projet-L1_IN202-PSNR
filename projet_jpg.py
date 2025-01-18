from PIL import Image
import numpy as np
# a du être changé car sinon cela ne marché pas dct et idc
# n'est pas reconnu
import scipy.fft as spfft
from math import log10, sqrt
# il faut l'installer pour save et lecture pour ouvrir une
# interface et choisir quel fichier on veut
from PyQt5.QtWidgets import QApplication, QFileDialog


def load(filename):
    '''Une fonction qui prend un path d'une image et
    qui le renvoie sous forme de matrice numpy'''
    toLoad = Image.open(filename)
    return np.asarray(toLoad)


def psnr(original, compressed):
    '''Calcule du psnr d'une image donnée et de sa
    forme compressé >= 40 image de bonne qualité'''
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def dct2(a):
    '''Renvoie la transformé de fournier d'une matrice carrée'''
    return spfft.dct(spfft.dct(a, axis=0, norm='ortho'), axis=1,
                     norm='ortho')


def idct2(a):
    '''Renvoie l'inverse de la transformé de fournier d'une matrice carrée'''
    return spfft.idct(spfft.idct(a, axis=0, norm='ortho'),  axis=1,
                      norm='ortho')


# Question 1
def YCbCr(mat):
    '''Prends en argument un matrice et la renvoie sous forme de
    cannaux Y Cb Cr sous forme de matrice de même dimension que la
    matrice d'origine'''
    # on copie chaque cannaux dans sa variable
    Red, Green, Blue = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]
    # on construit les cannaux Y Cb Cr
    Y = 0.299*Red + 0.587*Green + 0.114*Blue
    Cb = -0.1687*Red - 0.3313*Green + 0.5*Blue + 128
    Cr = 0.5*Red - 0.4187*Green - 0.0813*Blue + 128
    return Y, Cb, Cr


# Question 2
def RGB(Y, CB, CR):
    '''Prends en argument les cannaux Y Cb Cr, les trois sont de même
    dimension, sous forme de matrice et renvoie une matrice en rgb de même
    dimension que Y'''
    # on créer notre image en rgb sous forme de matrice rempli de 0
    out = np.empty((Y.shape[0], Y.shape[1], 3))
    # on rempli les cannaux R, G, B avec les formules appropier
    out[:, :, 0] = Y + 1.402*(CR - 128)
    out[:, :, 1] = Y - 0.34414*(CB - 128) - 0.71414*(CR - 128)
    out[:, :, 2] = Y + 1.772*(CB - 128)
    # on return cela sous une forme demandé [0, 255]
    return np.uint8(np.clip(out, 0, 255))


# Question 3 padding
def padding(mat, couleur):
    '''Prends en agurment une matrice de taille i,j et renvoi
    en matrice de taille m,n ou m et n sont divisible par 8 en
    remplissant les nouvelles colonnes et ou lignes par des 0
    Couleur sert a créer une image approprier si on est en rgb ou en ycbcr'''
    # on récupére la taille de mat
    height, width = mat.shape[:2]
    # on test si sa taille est divisible par 8 sinon on la renvoie
    # t'elle quel
    if width % 8 != 0 or height % 8 != 0:
        # on créer les nouvelles dimensions
        new_width = width
        new_height = height
        if width % 8 != 0:
            new_width = width + 8 - (width % 8)
        if height % 8 != 0:
            new_height = height + 8 - (height % 8)
        # on créer une nouvelles matrices avec les nouvelles dimension
        # selon le mode de couleur
        if couleur == "RGB":
            new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
        elif couleur == "YCBCR":
            new_image = Image.new('L', (new_width, new_height), 0)
            # on coopie notre ancien image dans la nouvelle
        new_image.paste(Image.fromarray(mat), (0, 0))
        new_image = np.array(new_image)
        return new_image
    return mat


# Question 3 reverse padding
def unpadding(pad_mat, base_shape):
    '''prends une matrice padé par la fonction padding() et renvoie
    la matrice sans se pad grace aux dimesnion d'origine donnée en argument'''
    height, width = base_shape[0], base_shape[1]
    return pad_mat[:height, :width]


# Question 4
def sous_echantillonnage_422(mat):
    '''Fonction qui prends une matrice rgb et qui la renvoie
    sous forme Y, Cb, Cr. Cb et Cr ayant perdu la moitier de leur
    colonne en fesant la moyenne de 2 pixel. les dimension de la matrice
    doivent être divisible par 2'''
    # on récupére la taille de mat
    height, width = mat.shape[:2]
    # on créer la nouvelle width de cb cr
    new_width = width // 2
    Y, cb, cr = YCbCr(mat)
    # on créer 2 nouvelle matrice pour cb cr
    new_cb = np.zeros((height, new_width))
    new_cr = np.zeros((height, new_width))
    # on copie dans nos nouvelle matrice la moyenne de 2 élément
    # côte à côte
    for i in range(height):
        for j in range(0, width, 2):
            # divise par sinon taille trop grande width taille origine
            # donc trop grand pour lew nouvelles matrice
            new_cb[i, j // 2] = (cb[i, j]//2 + cb[i, j + 1]//2)
            new_cr[i, j // 2] = (cr[i, j]//2 + cr[i, j + 1]//2)
    return Y, new_cb, new_cr


# Question 5
def echantillonnage_422(Y, cb, cr):
    '''Fonction qui prends Y, cb, cr
    (où cb cr ont des colonnes 2 fois plus petit que celle de Y)
    et qui renvoie Y, cb, cr de même dimension en répétant les pixels
    2 fois de cb cr'''
    # on récupére la taille de mat
    height, width = Y.shape[:2]
    # on créer des nouvelles matrice de taille de Y pour cb cr
    new_cb, new_cr = np.zeros((height, width)), np.zeros((height, width))
    # on recopie 2 fois chaque pixel de l'ancien matrice dans la nouvelle
    for i in range(height):
        for j in range(width):
            for _ in range(2):
                new_cb[i, j] = cb[i, j // 2]
                new_cr[i, j] = cr[i, j // 2]
    return Y, new_cb, new_cr


# Question 6
def block_8(mat, couleur):
    '''Prends en agurment une matrice de dimension m,n
    m,n étant divisible par 8 et couleur selon le si la matrice
    est en ycbcr et envoie une matrice composé dans block de 8x8 de
    l'image d'origine'''
    if couleur == "RGB":
        height, width = mat.shape[:2]
        # on créer une liste vide pour stoqué nos block
        block = []
        # on par notre matrice de 8 ligne en 8 ligne de même pour les colognes
        # pour créer nos block de 8x8
        for row in range(0, height, 8):
            for col in range(0, width, 8):
                case8 = mat[row:row+8, col:col+8, :]
                block.append(case8)
        return block
    elif couleur == 'YCBCR':
        height, width = mat.shape[:2]
        # on créer une matrice vide pour stoqué nos block
        block = np.empty((height//8 * width//8, 8, 8))
        index = 0
        # on par notre matrice de 8 ligne en 8 ligne de même pour les colognes
        # pour créer nos block de 8x8
        for row in range(0, height, 8):
            for col in range(0, width, 8):
                case8 = mat[row:row+8, col:col+8]
                block[index] = case8
                index += 1
        return block


# Question 7
def transforme_cos_discrete(mat):
    '''Prends une liste de block et applique a chaque block la
    transformation de fournier et renvoie une matrice de même dimension'''
    return np.array([dct2(row) for row in mat])


# Question 7
def transforme_inverse(mat):
    '''Prends une liste de block et applique a chaque block l'inverse de la
    transformation de fournier et renvoie une matrice de même dimension'''
    return np.array([idct2(row) for row in mat])


# Question 8
def seuil(mat, filtre, mode_couleur):
    '''Filtre les coefficiant de la matrice selon le
    filtre donnée en argument, mode_couleur : ycbcr pour le YCBCR
    rgb pour les block de RGB et RGBBlock pour une matrice rgb normale'''
    # si on est en rgb block
    if mode_couleur == "RGB":
        # on parcourt les block, lignes, cols, rgb, et filtre les coefs
        for block in range(len(mat)):
            for row in range(len(mat[block])):
                for col in range(len(mat[block][row])):
                    for rbg in range(len(mat[block][row][col])):
                        if mat[block][row][col][rbg] < filtre:
                            mat[block][row][col][rbg] = 0
    # si on est en ycbcr
    elif mode_couleur == "YCbCr":
        # on parcourt les block, lignes, cols et filtre les coefs
        for block in range(len(mat)):
            for row in range(len(mat[block])):
                for col in range(len(mat[block][row])):
                    if abs(mat[block][row][col]) < filtre:
                        mat[block][row][col] = 0
    # si on est en rgb
    elif mode_couleur == "RGBlock":
        # on parcourt les lignes, cols, rgb et filtre les coefs
        for row in range(len(mat)):
            for col in range(len(mat[row])):
                for rgb in range(len(mat[row][col])):
                    if mat[row][col][rgb] < filtre:
                        mat[row][col][rgb] = 0
    return mat


# fonction apart pour la Question 9
def decompo_ycbcr(mat):
    '''Prends une matrice de block de rgb et renvoie
    les listes correspondantes en Y, cb, cr'''
    Y, cb, cr = [], [], []
    for elt in mat:
        # on capture les équivalent des blocks et on ajoute aux liste
        # correspondantes
        a, b, c = YCbCr(elt)
        Y.append(a)
        cb.append(b)
        cr.append(c)
    return Y, cb, cr


# Question 11
def arrondi(mat):
    '''On arrondi les valeur de matrice et on la renvoie'''
    mat = np.round(mat).astype(np.int64)
    return mat


# Question 15 mode 3 coef
def coef_90(mat):
    '''Prends les 10% plus grands et plus le plus petit de ces
    derniers puis on applique seuil avec ce coef et donc renvoie une matrice
    avec 90% de 0'''
    tb_max = []
    # on prends 1/8 des valeur donc 12.5% des valeur les plus grandes
    for block in range(len(mat)):
        for row in range(len(mat[block])):
            tb_max.append(np.max(mat[block][row]))
    # on tri le tableau
    tb_max.sort()
    # on trouve le 10% de la taille de notre matrice
    dix_pourcent = - (len(mat) * 64 // 10) - 1
    # on trouve notre filtre
    min_coef = tb_max[dix_pourcent]
    # on return notre matrice avec seuil
    return seuil(mat, min_coef, "YCbCr")


# Question 16 exemple de tableau de quatification de wikipedia
quantization_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])


# Question 16
def quatificateur(mat):
    '''On divise nos block par notre table et on return'''
    return mat / quantization_table


# Question 16
def Unquantificateur(mat):
    '''On mutipli nos block par notre table et on return'''
    return mat * quantization_table


# Question 9
def compresse_Mode(mat, mode):
    '''Prends une matrice rgb et donne sa version compressé sous la forme
    [Y, cb, cr], mat.shape, mod'''
    # arrondi transformation de fournier ect et découpage en Y CB CR sont
    # présente pour save ce fichier compressé
    # mod 6/7 représente juste les mod 3/4 des question
    # les mod 3/4 sont l'ensemble des compression faite
    if mode == 0:
        # on pad notre matrice puis on la découpe en block de 8
        # puis en décompose nos block en Y cb cr
        Y, cb, cr = decompo_ycbcr(block_8(padding(mat, "RGB"), "RGB"))
        mod0 = [Y, cb, cr]
        # on utilise la transformé de fournier et on arrondi le tous
        for elt in range(3):
            mod0[elt] = arrondi(transforme_cos_discrete(mod0[elt]))
        return mod0, mat.shape, mode
    elif mode == 1:
        # comme mod0 mais on applique a nos bloque un seuil pour les
        # coefficient
        Y, cb, cr = decompo_ycbcr(seuil(block_8(padding(mat, "RGB"), "RGB"),
                                        20, "RGB"))
        mod1 = [Y, cb, cr]
        for elt in range(3):
            mod1[elt] = arrondi(transforme_cos_discrete(mod1[elt]))
        return mod1, mat.shape, mode
    elif mode == 2:
        # on appplique seuil a nos coef de notre mat
        mat = seuil(mat, 20, "RGBlock")
        # on sous echantillionne notre matrice
        Y, cb, cr = sous_echantillonnage_422(mat)
        mod2 = [Y, cb, cr]
        # on pad puis on transforme en block, puis transformé de fournier
        # puis on arrondi
        for couleur in range(3):
            mod2[couleur] = block_8(padding(mod2[couleur], "YCBCR"), "YCBCR")
            mod2[couleur] = arrondi(transforme_cos_discrete(mod2[couleur]))
        return mod2, mat.shape, mode
    elif mode == 3:
        # même explication qu'au mode 2 sauf qu'on applique avant la
        # transformation de fournier notre 90% de 0
        mat = seuil(mat, 20, "RGBlock")
        Y, cb, cr = sous_echantillonnage_422(mat)
        mod3 = [Y, cb, cr]
        for elt in range(3):
            mod3[elt] = block_8(padding(mod3[elt], "YCBCR"), "YCBCR")
            mod3[elt] = arrondi(transforme_cos_discrete(
                coef_90(mod3[elt])))
        return mod3, mat.shape, mode
    elif mode == 4:
        # de même que pour le mode 3 sauf qu'après la transformation de
        # fournier on lui applique notre table de quatiffication
        mat = seuil(mat, 20, "RGBlock")
        Y, cb, cr = sous_echantillonnage_422(mat)
        mod4 = [Y, cb, cr]
        for elt in range(3):
            mod4[elt] = block_8(padding(mod4[elt], "YCBCR"), "YCBCR")
            mod4[elt] = transforme_cos_discrete(coef_90(mod4[elt]))
            mod4[elt] = arrondi(quatificateur(mod4[elt]))
        return mod4, mat.shape, mode
    elif mode == 6:
        # on applique juste la compression du coef 90% de 0
        Y, cb, cr = decompo_ycbcr(block_8(padding(mat, "RGB"), "RGB"))
        mod6 = [Y, cb, cr]
        for elt in range(3):
            mod6[elt] = arrondi(transforme_cos_discrete(
                coef_90(mod6[elt])))
        return mod6, mat.shape, mode
    elif mode == 7:
        # on applique juste la compression de la table de quatification
        Y, cb, cr = decompo_ycbcr(block_8(padding(mat, "RGB"), "RGB"))
        mod7 = [Y, cb, cr]
        for elt in range(3):
            mod7[elt] = quatificateur(transforme_cos_discrete(mod7[elt]))
            mod7[elt] = arrondi(mod7[elt])
        return mod7, mat.shape, mode


# Question 12
def RLE_0(ligne):
    '''Prends une chaine de caractère et donne sa transformé
    par le mode rle uniquement sur les 0'''
    # on créer notre ligne rle
    new_ligne = ""
    # pour savoir si les 0 ne sont pas dans des nombres
    tb = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # on compte le nombre de 0
    stock = 0
    for elt in range(len(ligne)):
        # on teste si 0 n'est pas dans un nombre si ce n'est pas
        # le cas on compte +1 0
        if ligne[elt] == "0" and not (ligne[elt-1] in tb):
            stock += 1
        elif ligne[elt] == " ":
            # si on n'a pas écrit notre nouvelle ligne ou que le dernier
            # caractére est un espace ou que notre stock de 0 > 1 on ne fais
            # rien (sinon on écrirai des espaces indésirables)
            if new_ligne == "" or new_ligne[-1] == " " or stock > 0:
                pass
            # sinon on érit un espace
            else:
                new_ligne += " "
        else:
            if stock > 0:
                # fin de la ligne donc pas dépasse et on écrit les #k
                if elt+1 == len(ligne):
                    new_ligne += "#" + str(stock)
                # pas la fin de la ligne donc pas dépasse et on écrit les #k
                # + " "
                else:
                    new_ligne += "#" + str(stock) + " "
                stock = 0
            # écrit l'élément
            new_ligne += ligne[elt]
    # on enléce le ] et on le remplace par " "
    new_ligne = new_ligne[:-1] + " "
    return new_ligne


# Question 17 écriture
def zigzag_scan(matrice):
    '''Prends une matrice numpy 8x8 et la lis en diagonal
    et return un matrice numpy avec 64 valeur'''
    # variable utile
    zigzag, indice = [], [0, 0]
    place = 7
    RIGHT, DOWN = True, False
    # on parcour 15 notre block = perimitre - 1 du block
    for index in range(15):
        # pour ne pas modifier le premier indice
        condition = True
        # première moitier du block 8 premières lignes en diago
        if index <= 7:
            for _ in range(index+1):
                # on ne modifie pas l'indice que une fois par ligne
                if condition:
                    condition = False
                # tourné/descendu quand on arrive au bord du block
                # si on a tourné a droite on monte en ligne et descend en col
                elif RIGHT:
                    indice[0] -= 1
                    indice[1] += 1
                # si on a descendu on descend en ligne et on monte en col
                elif DOWN:
                    indice[0] += 1
                    indice[1] -= 1
                # on ajoute a notre nouvelle ligne la valeur
                zigzag.append(matrice[indice[0], indice[1]])
            # on veut pas modifier comme cela la valeur sinon on est
            # hors du tableau/matrice [8, 0]
            if index < 7:
                # on échange les valeurs logic des deux pour se déplacer
                # correctement dans notre tableau/matrice
                # si right on tourne a droite donc col + 1
                if RIGHT:
                    indice[1] += 1
                    RIGHT, DOWN = False, True
                # si down on descend donc row +1
                elif DOWN:
                    indice[0] += 1
                    RIGHT, DOWN = True, False
        # deuxième moitier du tableau (diago milieu pas comprise)
        else:
            # right et down ne sont pas cohérant a ce qu'il se passe
            # mais c'est pour matcher ce qu'on obtiens juste avant en
            # valeur logic donc même explication que juste audessus
            if RIGHT:
                indice[0] += 1
                RIGHT, DOWN = False, True
            elif DOWN:
                indice[1] += 1
                RIGHT, DOWN = True, False
                # ici on doit diminuer vu qu'on est dans la partie
                # triangulaire inférieur de la matrice donc place
                # commence a 7 jusqu'a 1
            for _ in range(place):
                # même explication sauf que la right et down comme on les a
                # inversé leurs nom est donc ambigu
                if condition:
                    condition = False
                elif RIGHT:
                    indice[0] -= 1
                    indice[1] += 1
                elif DOWN:
                    indice[0] += 1
                    indice[1] -= 1
                zigzag.append(matrice[indice[0], indice[1]])
            place -= 1
    return np.array(zigzag)


# Question 10
def save(block, dimension, mod, encoding):
    '''Prends une liste contenant les listes des blocks Y, cb, cr
    la dimension de la matrice d'origine, le mode de compression et son
    encoding et demande ou et dans quelle fichier on écrit notre compression'''
    # on demande où on veut save
    if encoding not in ["NORLE", "RLE", "ZIGZAGRLE"]:
        return
    _ = QApplication([])
    initial_dir = "./save"
    file_path, _ = QFileDialog.getSaveFileName(directory=initial_dir,
                                               filter="JPEG Image (*.jpg)")
    height, width = dimension[0], dimension[1]
    if file_path == "":
        return
    file = open(file_path, "w")
    # on écrit les informations demandé
    ecriture = 'JPG\n' + str(height) + " " + str(width) + "\n"
    ecriture += "mod" + str(mod) + "\n" + encoding
    file.write(ecriture)
    if encoding == "RLE":
        # on parcour notre liste contenat Y, cb, cr en block
        # on parcourt les blocks etr on les sépare par \n et on reset valeur
        # on parcourt les row et col des block on ajoute chaque valeur du block
        # sous forme de str à valeur puis on applique a valeur le code RLE
        # on écrit la ligne obtenu par le RLE
        for ycbcr in range(3):
            for blocks in range(len(block[ycbcr])):
                file.write("\n")
                valeur = ""
                for row in range(len(block[ycbcr][blocks])):
                    for value in range(len(block[ycbcr][blocks][row])):
                        valeur += str(block[ycbcr][blocks][row][value]) + " "
                valeur = valeur[:] + "]"
                file.write(RLE_0(valeur))
    elif encoding == "NORLE":
        # on parcour notre liste contenat Y, cb, cr en block
        # on parcourt les blocks etr on les sépare par \n et on reset valeur
        # on parcourt les row et col des block on écrit chaque valeur du block
        # suivit d'un espace  pour les séparé
        for ycbcr in range(3):
            for blocks in range(len(block[ycbcr])):
                file.write("\n")
                for row in range(len(block[ycbcr][blocks])):
                    for value in range(len(block[ycbcr][blocks][row])):
                        file.write(str(block[ycbcr][blocks][row][value]) + " ")
    elif encoding == "ZIGZAGRLE":
        # on parcour nos block avec les saut de ligne
        # on reset valeur a chaque block on lit les blocks
        # en zigzag puis on lit la ligne obtenue pour la mettre en
        # str pour l'écrire en rle
        for ycbcr in range(3):
            for blocks in range(len(block[ycbcr])):
                file.write("\n")
                valeur = ""
                new_ligne = zigzag_scan(block[ycbcr][blocks])
                for value in range(64):
                    valeur += str(new_ligne[value]) + " "
                valeur = valeur[:] + "]"
                file.write(RLE_0(valeur))
    file.close()


# Question 13
def decompression(tb_YCbCr, dimension, mod):
    '''Prends une liste de block de [Y, cb, cr] les dimension d'origine
    et le mode de compression utilisé pour renvoyer l'image en rgb sans
    compression'''
    # on regarde si mode est bien un mode qu'on a fait
    tb = [elt for elt in range(5)] + [6, 7]
    if mod in tb:
        height, width = dimension[0], dimension[1]
        # on a besoin de notre width paddé
        new_width = width
        if width % 8 != 0:
            new_width = width + (width % 8)
        # créer nos list pour contenir les canaux YCBCR de
        # taille height x width
        Y, Cb, Cr = [], [], []
        tb = [Y, Cb, Cr]
        # on enléve notre table de quantification selon le mode
        if mod == 4 or mod == 7:
            for ycbcr in range(3):
                tb_YCbCr[ycbcr] = Unquantificateur(tb_YCbCr[ycbcr])
        # on enlève la transformation de fournier
        for elt in range(3):
            tb_YCbCr[elt] = transforme_inverse(tb_YCbCr[elt])
        # on parcour notre liste contenat Y cb cr puis on parcour nos block
        # de new_width // 8 pour reavoir nos lignes on répète 8 fois pour
        # avoir nos 8 lignes des blocks on ajoute un tableau vide pour avoir
        # notre ligne complete puis on pacour new_width // 8 puis refaire nos
        # lignes et 8 fois pour nos 8 valeurs des blocks et on ajoute a nos
        # ligne nos colonne. Si on a un mode qui use sous_echan on faire
        # divisé par 2 pour cb cr new_width car ils le sont
        for ycbcr in range(3):
            if mod in [2, 3, 4] and ycbcr == 1:
                new_width = new_width // 2
            for row in range(0, len(tb_YCbCr[ycbcr]), new_width // 8):
                for huit in range(8):
                    tb[ycbcr].append([])
                    for col in range(new_width // 8):
                        for elt in range(8):
                            tb[ycbcr][len(tb[ycbcr])-1].append(tb_YCbCr[ycbcr]
                                                               [row + col]
                                                               [huit][elt])
        # on refait des matrices
        Y, Cb, Cr = np.array(Y), np.array(Cb), np.array(Cr)
        # si un mode est dans ce tableau => cb cr on 2 fois mon de
        # colonne que Y
        if mod in [2, 3, 4]:
            Y, Cb, Cr = echantillonnage_422(Y, Cb, Cr)
        # on refait notre matrice rgb
        img = RGB(Y, Cb, Cr)
        # on unpad
        img = unpadding(img, [height, width])
        return img


# Question 14 pour le RLE
def lecture_RLE(ligne):
    '''Prends une ligne de str binaire en RLE sur les 0
    et la redonne sous forme de liste de int'''
    # on refait notre ligne de valeur
    new_ligne = []
    # on parcourt notre ligne si notre valeur contient
    # un # en binaire => on le retransforme en str puis on enléve
    # le # pour répété le nombre de 0 indiqué sinon on ajoute la valeur
    # en la int()
    for elt in range(len(ligne)):
        value = ligne[elt]
        if b'#' in value:
            value = ligne[elt].decode('utf-8').strip()
            value = int(value[1:])
            for _ in range(value):
                new_ligne.append(0)
        else:
            value = int(value)
            new_ligne.append(value)
    return new_ligne


# Question 17 lecture
def lecture_zigzag(ligne):
    '''Lis un tableau de 64 valeur et le transforme
    en block de 8x8 en remplissant via les diagonals'''
    # on créer quelque variable utile
    block = np.zeros((8, 8))
    indice = [0, 0]
    place, lign_indice = 7, 0
    RIGHT, DOWN = True, False
    # le programme est exactement le même que zigzag_scan
    # sauf qu'ici on append rien on ligne notre ligne grace
    # a lign_indice et on lis block et on lui change ces valeurs
    # par celle de la ligne
    for index in range(15):
        condition = True
        if index <= 7:
            for _ in range(index+1):
                if condition:
                    condition = False
                elif RIGHT:
                    indice[0] -= 1
                    indice[1] += 1
                elif DOWN:
                    indice[0] += 1
                    indice[1] -= 1
                block[indice[0], indice[1]] = ligne[lign_indice]
                lign_indice += 1
            if index < 7:
                if RIGHT:
                    indice[1] += 1
                    RIGHT, DOWN = False, True
                elif DOWN:
                    indice[0] += 1
                    RIGHT, DOWN = True, False
        else:
            if RIGHT:
                indice[0] += 1
                RIGHT, DOWN = False, True
            elif DOWN:
                indice[1] += 1
                RIGHT, DOWN = True, False
            for _ in range(place):
                if condition:
                    condition = False
                elif RIGHT:
                    indice[0] -= 1
                    indice[1] += 1
                elif DOWN:
                    indice[0] += 1
                    indice[1] -= 1
                block[indice[0], indice[1]] = ligne[lign_indice]
                lign_indice += 1
            place -= 1
            # sinon on a des valeur a virgule exemple : x.
            block = np.round(block).astype(np.int64)
    return block


# Question 14
def lecture(path):
    '''Prends un path et le lis seulement si c'est un JPG
    et reforme les block Y cb cr et return le [Y, cb, cr], dimension
    ,mode'''
    stock = []
    file = open(path, "rb")
    if b'JPG' not in file.readline():
        return
    for _ in range(3):
        stock.append(file.readline())
    # on refait nos variable qu'on a besoin
    stock[0] = stock[0].split()
    height = int(stock[0][0])
    width = int(stock[0][1])
    new_height = (height + height % 8) // 8
    new_width = (width + width % 8) // 8
    Y_block, Cb_block, Cr_block = [], [], []
    tb_autre = [Y_block, Cb_block, Cr_block]
    encoding = stock[2].decode('utf-8').strip()
    mod = int(stock[1].decode('utf-8').strip()[-1])
    # Les deux lectures(NORLE RLE) sont les mêmes a la seule différence
    # quand NORLE on ajoute directment aux ligne int(valeur)
    # en RLE on a la ligne en int donc on la ligne normalement
    # Pour le reste on parcour le nombre de block selon le mode
    # on recréer notre block de 8x8 on lis notre ligne de 64
    # valeur on ajoute 8 valeur a une ligne puis en change de ligne
    # qu'a avoir fais les 8 lignes
    if encoding == "RLE":
        dim = new_width * new_height
        for indice in range(3):
            if mod in [2, 3, 4] and indice == 1:
                dim = dim // 2
            for _ in range(dim):
                tb_ligne = lecture_RLE(file.readline().split())
                add_tb = [[] for _ in range(8)]
                compte_ligne, compte_value = 0, 0
                for elt in range(len(tb_ligne)):
                    if compte_value == 8:
                        compte_value = 0
                        compte_ligne += 1
                    add_tb[compte_ligne].append(tb_ligne[elt])
                    compte_value += 1
                tb_autre[indice].append(add_tb)
    elif encoding == 'NORLE':
        dim = new_width * new_height
        for indice in range(3):
            if mod in [2, 3, 4] and indice == 1:
                dim = dim // 2
            for _ in range(dim):
                tb_ligne = file.readline().split()
                add_tb = [[] for _ in range(8)]
                compte_ligne, compte_value = 0, 0
                for elt in range(len(tb_ligne)):
                    if compte_value == 8:
                        compte_value = 0
                        compte_ligne += 1
                    tb_ligne[elt] = int(tb_ligne[elt])
                    add_tb[compte_ligne].append(tb_ligne[elt])
                    compte_value += 1
                tb_autre[indice].append(add_tb)
    elif encoding == "ZIGZAGRLE":
        # on lis nos block si on a certain mod on doit changer
        # les dimension. on reconstruit la ligne via lecture_RLE
        # vu qu'on a écrit en zigzag on appelle lecture_zigzag
        # qui grace a la ligne de RLE refait notre block et on l'ajoute
        # a notre tableau
        dim = new_width * new_height
        for indice in range(3):
            if mod in [2, 3, 4] and indice == 1:
                dim = dim // 2
            for _ in range(dim):
                tb_ligne = lecture_RLE(file.readline().split())
                tb_autre[indice].append(lecture_zigzag(tb_ligne))
    else:
        file.close()
        return
    file.close()
    return tb_autre, [height, width], mod


# pour test les fonctionnalité
def testfunction(id):
    if id == "RGB-YCbCr":
        Y, Cb, Cr = YCbCr(test)
        test2 = RGB(Y, Cb, Cr)
        return test2, psnr(test, test2)
    if id == "padding":
        return padding(test, "RGB")
    if id == "pad-unpad":
        # plante si on calcule le psnr vu qu'on obtient la même img
        return unpadding(padding(test, "RGB"), test.shape)
    if id == "echan_422":
        Y, Cb, Cr = sous_echantillonnage_422(test)
        Y, Cb, Cr = echantillonnage_422(Y, Cb, Cr)
        test2 = RGB(Y, Cb, Cr)
        return test2, psnr(test, test2)
    if id == "mod0":
        tbYCBCR, dim, mod = compresse_Mode(test, 0)
        save(tbYCBCR, dim, mod, encodade)
    if id == "mod1":
        tbYCBCR, dim, mod = compresse_Mode(test, 1)
        save(tbYCBCR, dim, mod, encodade)
    if id == "mod2":
        tbYCBCR, dim, mod = compresse_Mode(test, 2)
        save(tbYCBCR, dim, mod, encodade)
    if id == "mod3":
        tbYCBCR, dim, mod = compresse_Mode(test, 3)
        save(tbYCBCR, dim, mod, encodade)
    if id == "mod4":
        tbYCBCR, dim, mod = compresse_Mode(test, 4)
        save(tbYCBCR, dim, mod, encodade)
    if id == "mod6":
        tbYCBCR, dim, mod = compresse_Mode(test, 6)
        save(tbYCBCR, dim, mod, encodade)
    if id == "mod7":
        tbYCBCR, dim, mod = compresse_Mode(test, 7)
        save(tbYCBCR, dim, mod, encodade)
    if id == "lecture-decompresse":
        _ = QApplication([])
        initial_dir = "./save"
        file_dialog = QFileDialog(directory=initial_dir,
                                  filter="JPEG Image (*.jpg)")
        file_dialog.exec_()
        file_path = file_dialog.selectedFiles()[0]
        tb, dim, mode = lecture(file_path)
        test2 = decompression(tb, dim, mode)
        return test2, psnr(test, test2)


test = load("./test.png")
# NORLE, RLE, ZIGZAGRLE
encodade = "ZIGZAGRLE"
# modx (∀x ∈ [0, 1, 2, 3, 4, 6, 7])
testfunction("mod4")
# echan_422, RGB-YCBCR, lecture-decompresse
img, note = testfunction("lecture-decompresse")
print(note)
# padding, pad-unpad
# img = testfunction("padding")
Image.fromarray(img, 'RGB').show()
