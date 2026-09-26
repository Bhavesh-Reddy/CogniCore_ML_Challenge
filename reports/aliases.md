# Admin-region aliases (Step 2.1)

Learned from 300,000 sampled matched TRAIN pairs (seed 42), no external data. 61 canonical values (identity entries) + 72 variants = 133 entries in `work/aliases.json`.

Rules: variant count >= 50 and >= 80% of its occurrences; both canonical and variant must be the LAST comma-part in >= 80% of their records (this keeps states and drops cities and localities); canonical = the S1 form. See the aliases.py docstring.

**France:** there is no France data in train, so no France aliases were learned. French regions and departments (e.g. Hauts-de-France / Nord) stay as plain address tokens. Hand-written French street-type expansions (r -> rue, av -> avenue, bd -> boulevard, ...) are in normalize.py.

## Sanity check

- mh -> maharashtra: learned
- mharastr -> maharashtra: learned
- tx -> texas: not learned (tx maps to 'tx')
- texas -> tx: learned
- krnatk -> karnataka: learned
- dl -> delhi: learned

## Top 60 mappings by count, grouped by country

Country = the S1 country in most of the pairs behind the mapping.

### India

| variant | canonical | count | variant occurrences |
|---|---|---|---|
| mh | maharashtra | 7,085 | 8,069 |
| mharastr | maharashtra | 4,769 | 5,417 |
| dl | delhi | 4,673 | 5,168 |
| dilli | delhi | 3,028 | 3,366 |
| up | uttar pradesh | 2,858 | 3,206 |
| ka | karnataka | 2,456 | 2,831 |
| gj | gujarat | 2,191 | 2,462 |
| wb | west bengal | 2,176 | 2,433 |
| uttr prdes | uttar pradesh | 1,888 | 2,125 |
| krnatk | karnataka | 1,711 | 1,911 |
| tg | telangana | 1,696 | 1,905 |
| tmilnatu | tamil nadu | 1,583 | 1,821 |
| gujrat | gujarat | 1,444 | 1,626 |
| pscimbng | west bengal | 1,438 | 1,620 |
| hr | haryana | 1,383 | 1,582 |
| rj | rajasthan | 1,220 | 1,355 |
| telmgan | telangana | 1,110 | 1,287 |
| kl | kerala | 987 | 1,127 |
| mp | madhya pradesh | 888 | 996 |
| hriyana | haryana | 862 | 973 |
| br | bihar | 854 | 978 |
| rajsthan | rajasthan | 819 | 929 |
| keralam | kerala | 733 | 846 |
| kerlm | kerala | 690 | 789 |
| ap | andhra pradesh | 665 | 755 |
| mdhy prdes | madhya pradesh | 506 | 577 |
| pb | punjab | 470 | 534 |
| amdhrprdes | andhra pradesh | 418 | 478 |
| od | orissa | 410 | 464 |
| odisha | orissa | 399 | 463 |

### US

| variant | canonical | count | variant occurrences |
|---|---|---|---|
| texas | tx | 7,192 | 7,728 |
| north carolina | nc | 5,088 | 5,491 |
| ohio | oh | 4,601 | 4,954 |
| illinois | il | 4,043 | 4,344 |
| tennessee | tn | 3,327 | 3,591 |
| virginia | va | 3,217 | 3,455 |
| massachusetts | ma | 3,121 | 3,377 |
| arizona | az | 3,033 | 3,286 |
| indiana | in | 2,651 | 2,868 |
| maryland | md | 2,129 | 2,312 |
| california | ca | 1,948 | 2,137 |
| wisconsin | wi | 1,734 | 1,847 |
| alabama | al | 1,672 | 1,789 |
| minnesota | mn | 1,644 | 1,765 |
| kentucky | ky | 1,633 | 1,758 |
| oregon | or | 1,605 | 1,740 |
| arkansas | ar | 1,450 | 1,568 |
| missouri | mo | 1,315 | 1,424 |
| utah | ut | 1,250 | 1,341 |
| iowa | ia | 1,225 | 1,312 |
| oklahoma | ok | 1,194 | 1,283 |
| connecticut | ct | 975 | 1,050 |
| west virginia | wv | 874 | 932 |
| kansas | ks | 788 | 858 |
| maine | me | 659 | 716 |
| new mexico | nm | 627 | 670 |
| pennsylvania | pa | 579 | 628 |
| montana | mt | 551 | 588 |
| north dakota | nd | 355 | 380 |
| district of columbia | dc | 333 | 360 |

## Canonical values (identity entries), most frequent first

maharashtra (21,888), tx (15,681), delhi (14,109), ny (12,132), nc (11,072), oh (10,087), il (8,819), uttar pradesh (8,527), karnataka (7,851), tamil nadu (7,217), tn (7,042), va (7,031), az (6,754), ma (6,740), gujarat (6,690), west bengal (6,624), telangana (6,492), in (5,838), wa (4,967), md (4,673), ca (4,291), haryana (4,139), kerala (4,034), wi (3,816), rajasthan (3,706), al (3,669), mn (3,556), ky (3,497), or (3,470), ar (3,203), mo (3,019), ut (2,721), bihar (2,647), ia (2,634), ok (2,629), madhya pradesh (2,601), ct (2,121), andhra pradesh (1,981), wv (1,868), ks (1,790), orissa (1,629), me (1,407), punjab (1,406), nm (1,356), pa (1,352), mt (1,229), de (1,006), nd (770), dc (741), ne (629), vt (599), co (535), la (534), ak (438), ga (360), wy (338), sc (324), sd (155), id (124), ri (72), fl (70)

Elapsed 83.6s.
