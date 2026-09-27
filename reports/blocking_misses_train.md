# Blocking misses (train, val fold, shards 0-3)

Recall is below the 0.95 gate, so these go to the team before any key change. Produced by `python -m business_entity_resolution.src.blocking --split train --shards 0,1,2,3` (seed 42).

```
35,304 true val pairs missed in shards [0, 1, 2, 3]; 10 random examples:

  S1-921817676 (US)  <->  S3-42217455
    raw : 'Gulf College LLC' | '42145 Jay Bird Court, Kenai Peninsula Borough, AK'
    norm: core='gulf college' nospace='gulfcollege' addr_core='42145 jay bird court kenai peninsula borough' first_num='42145'
    raw : 'gulfcollege.com' | 'Jay Bird Ct, Soldotna, Alaska'
    norm: core='gulfcollege' nospace='gulfcollege' addr_core='jay bird court soldotna' first_num=''

  S1-331232853 (India)  <->  S3-47823711
    raw : 'Bangalore. Sanghi Ltd' | 'No.14, 6Th Floor, Naveen Complex, M S Road, Bangalore., Karnataka'
    norm: core='bangalore sanghi' nospace='bangaloresanghi' addr_core='number 14 6th floor naveen complex m south road bangalore' first_num='14'
    raw : 'Bangalore. Sanghi-Ltd.' | 'B3/14, Bangalore., KA'
    norm: core='bangalore sanghi' nospace='bangaloresanghi' addr_core='b3 14 bangalore' first_num='3'

  S1-626175271 (US)  <->  S2-120035100
    raw : 'Drive Partners' | '1165 Wheatland Court, Yorkville, IL'
    norm: core='drive partners' nospace='drivepartners' addr_core='1165 wheatland court yorkville' first_num='1165'
    raw : 'DRIVE PÁRTNERS' | ''
    norm: core='drive partners' nospace='drivepartners' addr_core='' first_num=''

  S1-260137936 (India)  <->  S2-243559285
    raw : 'Hitech Engineering Private Limited' | 'P.No. A-33, F.No. A-303, Radha Vihar, N S Road, Jaipur, Rajasthan'
    norm: core='hitech engineering' nospace='hitechengineering' addr_core='p number a 33 f number a 303 radha vihar north south road jaipur' first_num='33'
    raw : 'हाईटेक इंजीनियरिंग प्राइवेट लिमिटेड' | 'P.NO. A-3-3, JAIPUR, Rajasthan'
    norm: core='haitek imjiniyrimg' nospace='haitekimjiniyrimg' addr_core='p number a 3 3 jaipur' first_num='3'

  S1-225617269 (India)  <->  S3-731207839
    raw : 'Laxmi Finance Private Limited' | 'Shop No 41, Right Portion Ground Floor Blk-F Shopping Centre Mansarover Garden, Delhi, West Delhi, Delhi'
    norm: core='laxmi finance' nospace='laxmifinance' addr_core='shop number 41 right portion ground floor block f shopping centre mansarover garden delhi west delhi' first_num='41'
    raw : 'लक्ष्मी फाइनेंस प्राइवेट लिमिटेड' | 'No 41., Delhi, West Delhi, DL'
    norm: core='lksmi phainems' nospace='lksmiphainems' addr_core='number 41 delhi west delhi' first_num='41'

  S1-402776863 (India)  <->  S2-513352771
    raw : 'Butterfly Power Private Limited' | 'Flat No 102 Sanvi Meadows, Bhagya Lakshmi Nagar Phasi 2, Serilingampally, K.V.Rangareddy, Telangana'
    norm: core='butterfly power' nospace='butterflypower' addr_core='flat number 102 sanvi meadows bhagya lakshmi nagar phasi 2 serilingampally k v rangareddy' first_num='102'
    raw : 'Butterfly Power Limited Center' | ''
    norm: core='butterfly power' nospace='butterflypower' addr_core='' first_num=''

  S1-271491734 (India)  <->  S3-934195838
    raw : 'Future Exports Private Limited' | 'C/O Ramkavl Maurya, Budhave Ali Nagar, Maunath Bhanjan, Mau, Uttar Pradesh'
    norm: core='future exports' nospace='futureexports' addr_core='c o ramkavl maurya budhave ali nagar maunath bhanjan mau' first_num=''
    raw : 'फ्यूचर एक्सपोर्ट्स प्राइवेट लिमिटेड' | 'C/o Rakavl Maurya, Budhave Ali Nagar, Mau, Maunath Bhanjan, UP'
    norm: core='phyucr eksports' nospace='phyucreksports' addr_core='c o rakavl maurya budhave ali nagar mau maunath bhanjan' first_num=''

  S1-956114390 (US)  <->  S3-723868523
    raw : 'HA Priority LLC' | '701 Main Street, Unit LOT 54, Wellington, OH'
    norm: core='ha priority' nospace='hapriority' addr_core='701 main street unit lot 54 wellington' first_num='701'
    raw : 'Ha Priority Llc' | ''
    norm: core='ha priority' nospace='hapriority' addr_core='' first_num=''

  S1-635693500 (India)  <->  S3-832033885
    raw : 'New Delhi Developers Limited' | 'House No 508/1 3 No Floor, Ground Floor-Wz Village, New Delhi, West Delhi, Delhi'
    norm: core='new delhi developers' nospace='newdelhidevelopers' addr_core='house number 508 1 3 number floor ground floor wz village new delhi west delhi' first_num='508'
    raw : 'Umbrajaxzeph' | 'House No 508/1 3 No Floor, Ground Floor-wz Village, West Delhi, New Delhi, DL'
    norm: core='umbrajaxzeph' nospace='umbrajaxzeph' addr_core='house number 508 1 3 number floor ground floor wz village west delhi new delhi' first_num='508'

  S1-270152853 (India)  <->  S3-496456311
    raw : 'Star Care Private Limited' | 'Manikchand House Plot No, 100-101 D Kennedy Road, Pune City, Pune, Maharashtra'
    norm: core='star care' nospace='starcare' addr_core='manikchand house plot number 100 101 d kennedy road pune city pune' first_num='100'
    raw : 'स्टार केयर प्राइवेट लिमिटेड' | 'Block G-#582 Manikchand House Plot No, 100-101 D Kennedy Road, Pune, Pune City, महाराष्ट्र'
    norm: core='star keyr' nospace='starkeyr' addr_core='block g 582 manikchand house plot number 100 101 d kennedy road pune pune city' first_num='582'
```
