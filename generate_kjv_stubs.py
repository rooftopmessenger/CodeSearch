"""
generate_kjv_stubs.py — Write KJV NT stub text files (40-Matthew.txt … 66-Revelation.txt)
to the texts/ folder in the same column format used by the ELS engine:
    book_num  chapter  verse  word1 word2 ...

These stubs contain authentic KJV public-domain text for the opening chapter(s)
of each NT book, giving the build system real English ELS material to index.
Replace with a full KJV NT corpus to reach the expected ~2M+ total-letter count.

Run:   uv run generate_kjv_stubs.py
"""

from pathlib import Path

TEXTS_DIR = Path("texts")
TEXTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Content keyed by (file_num, filename, book_name)
# Format: list of (chapter, verse, text) tuples — text is raw KJV prose.
# ---------------------------------------------------------------------------
KJV_NT: list[tuple[int, str, str, list[tuple[int, int, str]]]] = [
    (40, "40-Matthew.txt", "Matthew", [
        (1,1,"The book of the generation of Jesus Christ the son of David the son of Abraham"),
        (1,2,"Abraham begat Isaac and Isaac begat Jacob and Jacob begat Judas and his brethren"),
        (1,3,"And Judas begat Phares and Zara of Thamar and Phares begat Esrom and Esrom begat Aram"),
        (1,4,"And Aram begat Aminadab and Aminadab begat Naasson and Naasson begat Salmon"),
        (1,5,"And Salmon begat Booz of Rachab and Booz begat Obed of Ruth and Obed begat Jesse"),
        (1,6,"And Jesse begat David the king and David the king begat Solomon of her that had been the wife of Urias"),
        (1,7,"And Solomon begat Roboam and Roboam begat Abia and Abia begat Asa"),
        (1,8,"And Asa begat Josaphat and Josaphat begat Joram and Joram begat Ozias"),
        (1,9,"And Ozias begat Joatham and Joatham begat Achaz and Achaz begat Ezekias"),
        (1,10,"And Ezekias begat Manasses and Manasses begat Amon and Amon begat Josias"),
        (1,11,"And Josias begat Jechonias and his brethren about the time they were carried away to Babylon"),
        (1,12,"And after they were brought to Babylon Jechonias begat Salathiel and Salathiel begat Zorobabel"),
        (1,13,"And Zorobabel begat Abiud and Abiud begat Eliakim and Eliakim begat Azor"),
        (1,14,"And Azor begat Sadoc and Sadoc begat Achim and Achim begat Eliud"),
        (1,15,"And Eliud begat Eleazar and Eleazar begat Matthan and Matthan begat Jacob"),
        (1,16,"And Jacob begat Joseph the husband of Mary of whom was born Jesus who is called Christ"),
        (1,17,"So all the generations from Abraham to David are fourteen generations and from David until the carrying away into Babylon are fourteen generations and from the carrying away into Babylon unto Christ are fourteen generations"),
        (1,18,"Now the birth of Jesus Christ was on this wise When as his mother Mary was espoused to Joseph before they came together she was found with child of the Holy Ghost"),
        (1,19,"Then Joseph her husband being a just man and not willing to make her a publick example was minded to put her away privily"),
        (1,20,"But while he thought on these things behold the angel of the Lord appeared unto him in a dream saying Joseph thou son of David fear not to take unto thee Mary thy wife for that which is conceived in her is of the Holy Ghost"),
        (1,21,"And she shall bring forth a son and thou shalt call his name JESUS for he shall save his people from their sins"),
        (1,22,"Now all this was done that it might be fulfilled which was spoken of the Lord by the prophet saying"),
        (1,23,"Behold a virgin shall be with child and shall bring forth a son and they shall call his name Emmanuel which being interpreted is God with us"),
        (1,24,"Then Joseph being raised from sleep did as the angel of the Lord had bidden him and took unto him his wife"),
        (1,25,"And knew her not till she had brought forth her firstborn son and he called his name JESUS"),
        (2,1,"Now when Jesus was born in Bethlehem of Judaea in the days of Herod the king behold there came wise men from the east to Jerusalem"),
        (2,2,"Saying Where is he that is born King of the Jews for we have seen his star in the east and are come to worship him"),
        (2,3,"When Herod the king had heard these things he was troubled and all Jerusalem with him"),
        (2,4,"And when he had gathered all the chief priests and scribes of the people together he demanded of them where Christ should be born"),
        (2,5,"And they said unto him In Bethlehem of Judaea for thus it is written by the prophet"),
        (5,3,"Blessed are the poor in spirit for theirs is the kingdom of heaven"),
        (5,4,"Blessed are they that mourn for they shall be comforted"),
        (5,5,"Blessed are the meek for they shall inherit the earth"),
        (5,6,"Blessed are they which do hunger and thirst after righteousness for they shall be filled"),
        (5,7,"Blessed are the merciful for they shall obtain mercy"),
        (5,8,"Blessed are the pure in heart for they shall see God"),
        (5,9,"Blessed are the peacemakers for they shall be called the children of God"),
        (5,10,"Blessed are they which are persecuted for righteousness sake for theirs is the kingdom of heaven"),
        (5,11,"Blessed are ye when men shall revile you and persecute you and shall say all manner of evil against you falsely for my sake"),
        (5,12,"Rejoice and be exceeding glad for great is your reward in heaven for so persecuted they the prophets which were before you"),
        (6,9,"After this manner therefore pray ye Our Father which art in heaven Hallowed be thy name"),
        (6,10,"Thy kingdom come Thy will be done in earth as it is in heaven"),
        (6,11,"Give us this day our daily bread"),
        (6,12,"And forgive us our debts as we forgive our debtors"),
        (6,13,"And lead us not into temptation but deliver us from evil For thine is the kingdom and the power and the glory for ever Amen"),
    ]),
    (41, "41-Mark.txt", "Mark", [
        (1,1,"The beginning of the gospel of Jesus Christ the Son of God"),
        (1,2,"As it is written in the prophets Behold I send my messenger before thy face which shall prepare thy way before thee"),
        (1,3,"The voice of one crying in the wilderness Prepare ye the way of the Lord make his paths straight"),
        (1,4,"John did baptize in the wilderness and preach the baptism of repentance for the remission of sins"),
        (1,5,"And there went out unto him all the land of Judaea and they of Jerusalem and were all baptized of him in the river of Jordan confessing their sins"),
        (1,14,"Now after that John was put in prison Jesus came into Galilee preaching the gospel of the kingdom of God"),
        (1,15,"And saying The time is fulfilled and the kingdom of God is at hand repent ye and believe the gospel"),
        (10,14,"But when Jesus saw it he was much displeased and said unto them Suffer the little children to come unto me and forbid them not for of such is the kingdom of God"),
        (12,30,"And thou shalt love the Lord thy God with all thy heart and with all thy soul and with all thy mind and with all thy strength this is the first commandment"),
        (12,31,"And the second is like namely this Thou shalt love thy neighbour as thyself There is none other commandment greater than these"),
    ]),
    (42, "42-Luke.txt", "Luke", [
        (1,1,"Forasmuch as many have taken in hand to set forth in order a declaration of those things which are most surely believed among us"),
        (1,2,"Even as they delivered them unto us which from the beginning were eyewitnesses and ministers of the word"),
        (1,3,"It seemed good to me also having had perfect understanding of all things from the very first to write unto thee in order most excellent Theophilus"),
        (1,4,"That thou mightest know the certainty of those things wherein thou hast been instructed"),
        (2,1,"And it came to pass in those days that there went out a decree from Caesar Augustus that all the world should be taxed"),
        (2,2,"And this taxing was first made when Cyrenius was governor of Syria"),
        (2,3,"And all went to be taxed every one into his own city"),
        (2,4,"And Joseph also went up from Galilee out of the city of Nazareth into Judaea unto the city of David which is called Bethlehem because he was of the house and lineage of David"),
        (2,5,"To be taxed with Mary his espoused wife being great with child"),
        (2,6,"And so it was that while they were there the days were accomplished that she should be delivered"),
        (2,7,"And she brought forth her firstborn son and wrapped him in swaddling clothes and laid him in a manger because there was no room for them in the inn"),
        (15,11,"And he said A certain man had two sons"),
        (15,12,"And the younger of them said to his father Father give me the portion of goods that falleth to me And he divided unto them his living"),
        (15,20,"And he arose and came to his father But when he was yet a great way off his father saw him and had compassion and ran and fell on his neck and kissed him"),
    ]),
    (43, "43-John.txt", "John", [
        (1,1,"In the beginning was the Word and the Word was with God and the Word was God"),
        (1,2,"The same was in the beginning with God"),
        (1,3,"All things were made by him and without him was not any thing made that was made"),
        (1,4,"In him was life and the life was the light of men"),
        (1,5,"And the light shineth in darkness and the darkness comprehended it not"),
        (1,14,"And the Word was made flesh and dwelt among us and we beheld his glory the glory as of the only begotten of the Father full of grace and truth"),
        (3,16,"For God so loved the world that he gave his only begotten Son that whosoever believeth in him should not perish but have everlasting life"),
        (3,17,"For God sent not his Son into the world to condemn the world but that the world through him might be saved"),
        (11,25,"Jesus said unto her I am the resurrection and the life he that believeth in me though he were dead yet shall he live"),
        (14,6,"Jesus saith unto him I am the way the truth and the life no man cometh unto the Father but by me"),
    ]),
    (44, "44-Acts.txt", "Acts", [
        (1,1,"The former treatise have I made O Theophilus of all that Jesus began both to do and teach"),
        (1,2,"Until the day in which he was taken up after that he through the Holy Ghost had given commandments unto the apostles whom he had chosen"),
        (2,1,"And when the day of Pentecost was fully come they were all with one accord in one place"),
        (2,2,"And suddenly there came a sound from heaven as of a rushing mighty wind and it filled all the house where they were sitting"),
        (2,3,"And there appeared unto them cloven tongues like as of fire and it sat upon each of them"),
        (2,4,"And they were all filled with the Holy Ghost and began to speak with other tongues as the Spirit gave them utterance"),
        (2,38,"Then Peter said unto them Repent and be baptized every one of you in the name of Jesus Christ for the remission of sins and ye shall receive the gift of the Holy Ghost"),
    ]),
    (45, "45-Romans.txt", "Romans", [
        (1,1,"Paul a servant of Jesus Christ called to be an apostle separated unto the gospel of God"),
        (1,2,"Which he had promised afore by his prophets in the holy scriptures"),
        (1,16,"For I am not ashamed of the gospel of Christ for it is the power of God unto salvation to every one that believeth to the Jew first and also to the Greek"),
        (3,23,"For all have sinned and come short of the glory of God"),
        (6,23,"For the wages of sin is death but the gift of God is eternal life through Jesus Christ our Lord"),
        (8,28,"And we know that all things work together for good to them that love God to them who are the called according to his purpose"),
        (8,38,"For I am persuaded that neither death nor life nor angels nor principalities nor powers nor things present nor things to come"),
        (8,39,"Nor height nor depth nor any other creature shall be able to separate us from the love of God which is in Christ Jesus our Lord"),
        (12,1,"I beseech you therefore brethren by the mercies of God that ye present your bodies a living sacrifice holy acceptable unto God which is your reasonable service"),
    ]),
    (46, "46-1Corinthians.txt", "I Corinthians", [
        (1,1,"Paul called to be an apostle of Jesus Christ through the will of God and Sosthenes our brother"),
        (1,2,"Unto the church of God which is at Corinth to them that are sanctified in Christ Jesus called to be saints with all that in every place call upon the name of Jesus Christ our Lord both theirs and ours"),
        (13,1,"Though I speak with the tongues of men and of angels and have not charity I am become as sounding brass or a tinkling cymbal"),
        (13,4,"Charity suffereth long and is kind charity envieth not charity vaunteth not itself is not puffed up"),
        (13,12,"For now we see through a glass darkly but then face to face now I know in part but then shall I know even as also I am known"),
        (13,13,"And now abideth faith hope charity these three but the greatest of these is charity"),
        (15,3,"For I delivered unto you first of all that which I also received how that Christ died for our sins according to the scriptures"),
        (15,4,"And that he was buried and that he rose again the third day according to the scriptures"),
    ]),
    (47, "47-2Corinthians.txt", "II Corinthians", [
        (1,1,"Paul an apostle of Jesus Christ by the will of God and Timothy our brother unto the church of God which is at Corinth with all the saints which are in all Achaia"),
        (1,3,"Blessed be God even the Father of our Lord Jesus Christ the Father of mercies and the God of all comfort"),
        (4,17,"For our light affliction which is but for a moment worketh for us a far more exceeding and eternal weight of glory"),
        (5,17,"Therefore if any man be in Christ he is a new creature old things are passed away behold all things are become new"),
        (12,9,"And he said unto me My grace is sufficient for thee for my strength is made perfect in weakness Most gladly therefore will I rather glory in my infirmities that the power of Christ may rest upon me"),
    ]),
    (48, "48-Galatians.txt", "Galatians", [
        (1,1,"Paul an apostle not of men neither by man but by Jesus Christ and God the Father who raised him from the dead"),
        (1,2,"And all the brethren which are with me unto the churches of Galatia"),
        (2,20,"I am crucified with Christ nevertheless I live yet not I but Christ liveth in me and the life which I now live in the flesh I live by the faith of the Son of God who loved me and gave himself for me"),
        (5,22,"But the fruit of the Spirit is love joy peace longsuffering gentleness goodness faith"),
        (5,23,"Meekness temperance against such there is no law"),
        (6,7,"Be not deceived God is not mocked for whatsoever a man soweth that shall he also reap"),
    ]),
    (49, "49-Ephesians.txt", "Ephesians", [
        (1,1,"Paul an apostle of Jesus Christ by the will of God to the saints which are at Ephesus and to the faithful in Christ Jesus"),
        (2,8,"For by grace are ye saved through faith and that not of yourselves it is the gift of God"),
        (2,9,"Not of works lest any man should boast"),
        (6,11,"Put on the whole armour of God that ye may be able to stand against the wiles of the devil"),
        (6,12,"For we wrestle not against flesh and blood but against principalities against powers against the rulers of the darkness of this world against spiritual wickedness in high places"),
    ]),
    (50, "50-Philippians.txt", "Philippians", [
        (1,1,"Paul and Timotheus the servants of Jesus Christ to all the saints in Christ Jesus which are at Philippi with the bishops and deacons"),
        (4,4,"Rejoice in the Lord alway and again I say Rejoice"),
        (4,7,"And the peace of God which passeth all understanding shall keep your hearts and minds through Christ Jesus"),
        (4,13,"I can do all things through Christ which strengtheneth me"),
    ]),
    (51, "51-Colossians.txt", "Colossians", [
        (1,1,"Paul an apostle of Jesus Christ by the will of God and Timotheus our brother"),
        (1,15,"Who is the image of the invisible God the firstborn of every creature"),
        (1,16,"For by him were all things created that are in heaven and that are in earth visible and invisible whether they be thrones or dominions or principalities or powers all things were created by him and for him"),
        (3,2,"Set your affection on things above not on things on the earth"),
    ]),
    (52, "52-1Thessalonians.txt", "I Thessalonians", [
        (1,1,"Paul and Silvanus and Timotheus unto the church of the Thessalonians which is in God the Father and in the Lord Jesus Christ Grace be unto you and peace from God our Father and the Lord Jesus Christ"),
        (4,13,"But I would not have you to be ignorant brethren concerning them which are asleep that ye sorrow not even as others which have no hope"),
        (5,16,"Rejoice evermore"),
        (5,17,"Pray without ceasing"),
        (5,18,"In every thing give thanks for this is the will of God in Christ Jesus concerning you"),
    ]),
    (53, "53-2Thessalonians.txt", "II Thessalonians", [
        (1,1,"Paul and Silvanus and Timotheus unto the church of the Thessalonians in God our Father and the Lord Jesus Christ"),
        (1,2,"Grace unto you and peace from God our Father and the Lord Jesus Christ"),
        (3,10,"For even when we were with you this we commanded you that if any would not work neither should he eat"),
    ]),
    (54, "54-1Timothy.txt", "I Timothy", [
        (1,1,"Paul an apostle of Jesus Christ by the commandment of God our Saviour and Lord Jesus Christ which is our hope"),
        (2,5,"For there is one God and one mediator between God and men the man Christ Jesus"),
        (6,6,"But godliness with contentment is great gain"),
        (6,10,"For the love of money is the root of all evil which while some coveted after they have erred from the faith and pierced themselves through with many sorrows"),
    ]),
    (55, "55-2Timothy.txt", "II Timothy", [
        (1,1,"Paul an apostle of Jesus Christ by the will of God according to the promise of life which is in Christ Jesus"),
        (1,7,"For God hath not given us the spirit of fear but of power and of love and of a sound mind"),
        (2,15,"Study to shew thyself approved unto God a workman that needeth not to be ashamed rightly dividing the word of truth"),
        (3,16,"All scripture is given by inspiration of God and is profitable for doctrine for reproof for correction for instruction in righteousness"),
        (4,7,"I have fought a good fight I have finished my course I have kept the faith"),
    ]),
    (56, "56-Titus.txt", "Titus", [
        (1,1,"Paul a servant of God and an apostle of Jesus Christ according to the faith of God's elect and the acknowledging of the truth which is after godliness"),
        (2,11,"For the grace of God that bringeth salvation hath appeared to all men"),
        (3,5,"Not by works of righteousness which we have done but according to his mercy he saved us by the washing of regeneration and renewing of the Holy Ghost"),
    ]),
    (57, "57-Philemon.txt", "Philemon", [
        (1,1,"Paul a prisoner of Jesus Christ and Timothy our brother unto Philemon our dearly beloved and fellowlabourer"),
        (1,2,"And to our beloved Apphia and Archippus our fellowsoldier and to the church in thy house"),
        (1,10,"I beseech thee for my son Onesimus whom I have begotten in my bonds"),
        (1,16,"Not now as a servant but above a servant a brother beloved specially to me but how much more unto thee both in the flesh and in the Lord"),
    ]),
    (58, "58-Hebrews.txt", "Hebrews", [
        (1,1,"God who at sundry times and in divers manners spake in time past unto the fathers by the prophets"),
        (1,2,"Hath in these last days spoken unto us by his Son whom he hath appointed heir of all things by whom also he made the worlds"),
        (1,3,"Who being the brightness of his glory and the express image of his person and upholding all things by the word of his power when he had by himself purged our sins sat down on the right hand of the Majesty on high"),
        (11,1,"Now faith is the substance of things hoped for the evidence of things not seen"),
        (11,6,"But without faith it is impossible to please him for he that cometh to God must believe that he is and that he is a rewarder of them that diligently seek him"),
        (12,1,"Wherefore seeing we also are compassed about with so great a cloud of witnesses let us lay aside every weight and the sin which doth so easily beset us and let us run with patience the race that is set before us"),
    ]),
    (59, "59-James.txt", "James", [
        (1,1,"James a servant of God and of the Lord Jesus Christ to the twelve tribes which are scattered abroad greeting"),
        (1,5,"If any of you lack wisdom let him ask of God that giveth to all men liberally and upbraideth not and it shall be given him"),
        (1,22,"But be ye doers of the word and not hearers only deceiving your own selves"),
        (2,14,"What doth it profit my brethren though a man say he hath faith and have not works can faith save him"),
        (2,26,"For as the body without the spirit is dead so faith without works is dead also"),
    ]),
    (60, "60-1Peter.txt", "I Peter", [
        (1,1,"Peter an apostle of Jesus Christ to the strangers scattered throughout Pontus Galatia Cappadocia Asia and Bithynia"),
        (1,2,"Elect according to the foreknowledge of God the Father through sanctification of the Spirit unto obedience and sprinkling of the blood of Jesus Christ Grace unto you and peace be multiplied"),
        (3,15,"But sanctify the Lord God in your hearts and be ready always to give an answer to every man that asketh you a reason of the hope that is in you with meekness and fear"),
        (5,7,"Casting all your care upon him for he careth for you"),
    ]),
    (61, "61-2Peter.txt", "II Peter", [
        (1,1,"Simon Peter a servant and an apostle of Jesus Christ to them that have obtained like precious faith with us through the righteousness of God and our Saviour Jesus Christ"),
        (1,20,"Knowing this first that no prophecy of the scripture is of any private interpretation"),
        (1,21,"For the prophecy came not in old time by the will of man but holy men of God spake as they were moved by the Holy Ghost"),
        (3,9,"The Lord is not slack concerning his promise as some men count slackness but is longsuffering to us ward not willing that any should perish but that all should come to repentance"),
    ]),
    (62, "62-1John.txt", "I John", [
        (1,1,"That which was from the beginning which we have heard which we have seen with our eyes which we have looked upon and our hands have handled of the Word of life"),
        (1,5,"This then is the message which we have heard of him and declare unto you that God is light and in him is no darkness at all"),
        (4,8,"He that loveth not knoweth not God for God is love"),
        (4,16,"And we have known and believed the love that God hath to us God is love and he that dwelleth in love dwelleth in God and God in him"),
        (5,3,"For this is the love of God that we keep his commandments and his commandments are not grievous"),
    ]),
    (63, "63-2John.txt", "II John", [
        (1,1,"The elder unto the elect lady and her children whom I love in the truth and not I only but also all they that have known the truth"),
        (1,3,"Grace be with you mercy and peace from God the Father and from the Lord Jesus Christ the Son of the Father in truth and love"),
        (1,6,"And this is love that we walk after his commandments This is the commandment That as ye have heard from the beginning ye should walk in it"),
    ]),
    (64, "64-3John.txt", "III John", [
        (1,1,"The elder unto the wellbeloved Gaius whom I love in the truth"),
        (1,2,"Beloved I wish above all things that thou mayest prosper and be in health even as thy soul prospereth"),
        (1,4,"I have no greater joy than to hear that my children walk in truth"),
        (1,11,"Beloved follow not that which is evil but that which is good He that doeth good is of God but he that doeth evil hath not seen God"),
    ]),
    (65, "65-Jude.txt", "Jude", [
        (1,1,"Jude the servant of Jesus Christ and brother of James to them that are sanctified by God the Father and preserved in Jesus Christ and called"),
        (1,2,"Mercy unto you and peace and love be multiplied"),
        (1,3,"Beloved when I gave all diligence to write unto you of the common salvation it was needful for me to write unto you and exhort you that ye should earnestly contend for the faith which was once delivered unto the saints"),
        (1,24,"Now unto him that is able to keep you from falling and to present you faultless before the presence of his glory with exceeding joy"),
        (1,25,"To the only wise God our Saviour be glory and majesty dominion and power both now and ever Amen"),
    ]),
    (66, "66-Revelation.txt", "Revelation", [
        (1,1,"The Revelation of Jesus Christ which God gave unto him to shew unto his servants things which must shortly come to pass and he sent and signified it by his angel unto his servant John"),
        (1,3,"Blessed is he that readeth and they that hear the words of this prophecy and keep those things which are written therein for the time is at hand"),
        (1,8,"I am Alpha and Omega the beginning and the ending saith the Lord which is and which was and which is to come the Almighty"),
        (3,20,"Behold I stand at the door and knock if any man hear my voice and open the door I will come in to him and will sup with him and he with me"),
        (21,1,"And I saw a new heaven and a new earth for the first heaven and the first earth were passed away and there was no more sea"),
        (21,4,"And God shall wipe away all tears from their eyes and there shall be no more death neither sorrow nor crying neither shall there be any more pain for the former things are passed away"),
        (22,13,"I am Alpha and Omega the beginning and the end the first and the last"),
        (22,20,"He which testifieth these things saith Surely I come quickly Amen Even so come Lord Jesus"),
        (22,21,"The grace of our Lord Jesus Christ be with you all Amen"),
    ]),
]

total_files = 0
total_letters = 0

for book_num, fname, book_name, verses in KJV_NT:
    out_path = TEXTS_DIR / fname
    lines = []
    for ch, vs, text in verses:
        lines.append(f"{book_num} {ch} {vs} {text}")
    content = "\n".join(lines) + "\n"
    out_path.write_text(content, encoding="utf-8")
    letters = sum(1 for c in content if c.isalpha())
    total_letters += letters
    total_files += 1
    print(f"  Wrote {fname}  ({letters:,} alpha letters)")

print(f"\nDone: {total_files} files, {total_letters:,} total NT stub letters")
print("Note: These are opening-chapter stubs only.")
print("Replace with the full KJV NT corpus to reach ~790,000+ NT letters (2M+ combined).")
