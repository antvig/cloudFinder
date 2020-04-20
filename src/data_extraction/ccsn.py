from zipfile import ZipFile
from tqdm import tqdm
import shutil



def extract_ccsn(zip_in , img_out)
    ccsn_zip = ZipFile(zip_in + 'CCSN.zip')

    res = []

    for img in tqdm(ccsn_zipt.filelist):
        if f.filename[-3:]=='jpg':
            
            res.append([id, f.filename])

    with ZipFile(PATH + 'CCSN.zip') as ccsn_zip:
        for member in ccsn_zip.namelist():
            filename = member.spit('/')[-1]
            if member[-3:] !='jpg':
                continue

            # copy file (taken from zipfile's extract)
            source = ccsn_zip.open(member)
            target = open(img_out + filename, "wb")
            with source, target:
                shutil.copyfileobj(source, target)
            
            unique_id = 'cc_' + datetime.now().date().isoformat().replace('-','') +'_'+ str(uuid.uuid4()).split('-')[0]
            res.appen([unique_id, member])
    
    df = pd.DataFrame(res, columns=["id", "path"])
    df.to_csv(img_out + 'ccsn.csv', index=False)

    return df
                
if __name__ == '__main__':
    
    zip_in = '../../data/row/'
    img_out = '../../data/img/'
    
    extract_ccsn(zip_in, img_out)