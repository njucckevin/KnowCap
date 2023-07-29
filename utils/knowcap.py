import json
import random

landmarks = ['milan cathedral', 'beijing national stadium', 'oriental pearl', 'captiol hill', 'white house', 'eiffel tower', 'grand canyon', 'taj mahal', 'statue of liberty', 'golden gate bridge', 'buckingham palace', 'times square', 'empire state building', 'tower bridge', 'forbidden city', 'angkor wat', 'niagara falls', 'brooklyn bridge', 'big ben', 'red square', 'burj khalifa', 'machu picchu', 'colosseum', 'mount everest', 'sydney opera house', 'great wall of china', 'tower of london', 'yellowstone national park', 'alhambra', 'kremlin', 'vatican city', 'christ the redeemer', 'pyramids of giza', 'london eye', 'cn tower', 'westminster abbey', 'notre dame cathedral', 'leaning tower of pisa', 'zion national park', 'louvre museum', 'palace of versailles', 'parthenon', 'sydney harbour bridge', 'grand palace', 'mount rushmore', 'mount fuji', 'hagia sophia', 'edinburgh castle', 'petra', 'stonehenge', 'neuschwanstein castle', 'burj al arab', 'trevi fountain', 'space needle', 'uluru', 'chichen itza', 'berlin wall', 'willis tower', 'cliffs of moher', 'hollywood sign', 'golden gate park', 'palace of westminster', 'tokyo tower', 'hoover dam', 'vatican museums', 'brandenburg gate', 'wailing wall', 'mount kilimanjaro', 'taipei 101', 'sagrada familia', 'potala palace', 'mont saint michel', 'acropolis of athens', 'terracotta army', 'tiananmen square', 'summer palace', 'the bund', 'temple of heaven', 'shanghai tower', 'west lake', 'shaolin temple', 'tokyo skytree', 'shibuya crossing', 'sensoji temple', 'meiji shrine', 'british museum', 'windsor castle', 'victoria peak', 'victoria harbour']
print("Num of landmarks: "+str(len(landmarks)))
brands = ['disneyland', 'china southern airlines', 'nokia', 'cadillac', 'fifa world cup', 'iphone', 'ford', 'apple', 'shell', 'honda', 'nike', 'toyota', 'bmw', 'google', 'chevrolet', 'jeep', 'porsche', 'audi', 'dodge', 'sony', 'chanel', 'adidas', 'facebook', 'gucci', 'microsoft', 'ferrari', 'youtube', 'lamborghini', 'mercedes-benz', 'burberry', 'tesla', 'walmart', 'rolex', 'samsung', 'starbucks', 'yamaha', 'bentley', 'canon', 'louis vuitton', 'nikon', 'coca cola', 'huawei', 'cartier', 'pepsi', 'lg', 'aston martin', 'kawasaki', 'harley-davidson', 'dell', '3m', 'rolls-royce', 'burger king', 'hsbc', 'ibm', 'philips', 'mcdonald', 'intel', 'logitech', 'kfc', 'acer', 'bank of america', 'panasonic', 'razer', 'bose', 'mastercard', 'american express', "domino's", 'general electric', 'lenovo', 'oppo', 'intel core']
print("Num of brands: "+str(len(brands)))
foods = ['sushi', 'tacos', 'ramen', 'fried rice', 'white wine', 'dumplings', 'pho', 'kebab', 'biryani', 'kimchi', 'paella', 'smoked salmon', 'pad thai', 'croissants', 'fish and chips', 'satay', 'shawarma', 'baklava', 'hot pot', 'sashimi', 'empanadas', 'tiramisu', 'laksa', 'fajitas', 'churros', 'pierogi', 'soba noodles', 'peking duck', 'caviar', 'udon', 'tempura', 'miso soup', 'escargot', 'kung pao chicken', 'foie gras']
print("Num of foods: "+str(len(foods)))
charas = ['batman', 'barbie', 'joker', 'superman', 'santa claus', 'spider-man', 'captain america', 'iron man', 'sonic the hedgehog', 'cinderella', 'super mario', 'mickey mouse', 'jasmine', 'ariel', 'the grinch', 'charlie brown', 'godzilla', 'simba', 'woody', 'aladdin', 'optimus prime', 'the tramp', 'rapunzel', 'pikachu', 'shrek', 'spongebob squarepants', 'olaf', 'kermit the frog', 'pinocchio', 'monkey king', 'bambi', 'mulan', 'popeye', 'bugs bunny', 'tiana', 'king kong', 'gandalf', 'merida', 'minnie mouse', 'ursula', 'big bird', 'scooby-doo', 'homer simpson', 'buzz lightyear', 'winnie-the-pooh']
print("Num of charas: "+str(len(charas)))

#together = landmarks+brands+foods+charas
#print(len(together))
categories = ['fifa world cup', 'captiol hill', 'cadillac', 'oriental pearl', 'beijing national stadium', 'milan cathedral', 'nokia', 'china southern airlines', 'biryani', 'tiananmen square', 'potala palace', 'minnie mouse', 'great wall of china', 'adidas', 'louvre museum', 'kawasaki', 'toyota', 'ibm', 'baklava', 'hagia sophia', 'simba', 'golden gate park', 'tokyo skytree', 'pho', 'gucci', 'superman', 'mulan', 'cinderella', 'mercedes-benz', 'acer', 'grand palace', 'rapunzel', 'parthenon', 'logitech', 'batman', "domino's", 'walmart', 'willis tower', 'vatican museums', 'wailing wall', 'lg', 'ariel', 'tesla', 'huawei', 'victoria harbour', 'temple of heaven', 'caviar', 'white wine', 'burj khalifa', 'harley-davidson', 'lamborghini', 'tiana', 'charlie brown', 'golden gate bridge', 'spongebob squarepants', 'cn tower', 'the grinch', 'coca cola', 'optimus prime', 'chichen itza', 'machu picchu', 'vatican city', 'meiji shrine', 'burj al arab', 'bmw', 'honda', 'hsbc', 'chevrolet', 'times square', 'ford', 'escargot', 'pepsi', 'kung pao chicken', 'big bird', 'fajitas', 'statue of liberty', 'ursula', 'fish and chips', 'aladdin', 'intel core', 'victoria peak', 'uluru', 'jeep', 'ramen', 'forbidden city', 'mont saint michel', 'iron man', 'bugs bunny', 'grand canyon', 'eiffel tower', 'google', 'brandenburg gate', 'white house', 'ferrari', 'intel', 'kremlin', 'buzz lightyear', 'pyramids of giza', 'apple', 'yellowstone national park', 'tokyo tower', 'oppo', 'scooby-doo', 'mount rushmore', 'soba noodles', 'kimchi', 'buckingham palace', 'mount fuji', 'croissants', 'alhambra', 'pikachu', 'tower bridge', 'nike', 'tower of london', 'christ the redeemer', 'the tramp', 'youtube', 'woody', 'palace of versailles', 'super mario', 'barbie', 'cliffs of moher', 'disneyland', 'acropolis of athens', 'tempura', 'captain america', 'windsor castle', 'facebook', 'iphone', 'sonic the hedgehog', 'dumplings', 'terracotta army', 'shawarma', 'kermit the frog', 'miso soup', 'merida', 'spider-man', 'fried rice', 'yamaha', 'stonehenge', 'audi', 'mount everest', 'monkey king', 'canon', 'rolex', '3m', 'space needle', 'west lake', 'notre dame cathedral', 'neuschwanstein castle', 'paella', 'bentley', 'niagara falls', 'trevi fountain', 'chanel', 'taj mahal', 'sagrada familia', 'panasonic', 'popeye', 'winnie-the-pooh', 'dell', 'tiramisu', 'berlin wall', 'olaf', 'british museum', 'foie gras', 'pierogi', 'zion national park', 'lenovo', 'peking duck', 'nikon', 'shrek', 'red square', 'bambi', 'aston martin', 'mastercard', 'brooklyn bridge', 'petra', 'westminster abbey', 'dodge', 'philips', 'starbucks', 'shell', 'hollywood sign', 'samsung', 'the bund', 'burberry', 'bank of america', 'american express', 'louis vuitton', 'churros', 'joker', 'shibuya crossing', 'king kong', 'udon', 'kebab', 'empire state building', 'mount kilimanjaro', 'sashimi', 'sushi', 'tacos', 'sydney opera house', 'rolls-royce', 'jasmine', 'microsoft', 'empanadas', 'shaolin temple', 'godzilla', 'pad thai', 'london eye', 'sensoji temple', 'edinburgh castle', 'summer palace', 'sony', 'porsche', 'pinocchio', 'gandalf', 'palace of westminster', 'big ben', 'angkor wat', 'general electric', 'hot pot', 'santa claus', 'laksa', 'burger king', 'colosseum', 'shanghai tower', 'bose', 'homer simpson', 'taipei 101', 'kfc', 'razer', 'satay', 'mcdonald', 'hoover dam', 'mickey mouse', 'sydney harbour bridge', 'leaning tower of pisa', 'cartier', 'smoked salmon']
print(len(categories))
categories_to_keywords = {'fifa world cup': ['fifa world cup', 'world cup'], 'captiol hill': ['u.s. capitol', 'captiol hill'], 'cadillac': ['cadillac'], 'oriental pearl': ['shanghai television tower', 'oriental pearl'], 'beijing national stadium': ['national stadium', 'chinese national stadium', "bird's nest", 'beijing national stadium'], 'milan cathedral': ['milan cathedral', 'duomo di milano'], 'nokia': ['nokia'], 'china southern airlines': ['china southern airlines'], 'biryani': ['biryani'], 'tiananmen square': ['tiananmen', "tian'anmen", 'tiananmen square'], 'potala palace': ['potala palace'], 'minnie mouse': ['minnie mouse'], 'great wall of china': ['great wall of china'], 'adidas': ['adidas', 'adi dassler'], 'louvre museum': ['louvre', 'louvre museum'], 'kawasaki': ['kawasaki', 'khi'], 'toyota': ['toyota'], 'ibm': ['ibm', 'international business machines', 'big blue'], 'baklava': ['baklava'], 'hagia sophia': ['hagia sophia', 'holy wisdom', 'sophia of rome'], 'simba': ['simba', 'the lion king'], 'golden gate park': ['golden gate park', 'san francisco city park'], 'tokyo skytree': ['tokyo skytree'], 'pho': ['pho', 'vietnamese soup dish'], 'gucci': ['gucci'], 'superman': ['superman'], 'mulan': ['mulan'], 'cinderella': ['cinderella'], 'mercedes-benz': ['mercedes', 'benz', 'mercedes-benz'], 'acer': ['acer'], 'grand palace': ['grand palace', 'phra borom maha ratcha wang'], 'rapunzel': ['rapunzel'], 'parthenon': ['temple of athena', 'parthenon', 'athenian acropolis'], 'logitech': ['logi', 'logitech'], 'batman': ['batman'], "domino's": ["domino's"], 'walmart': ['walmart', 'wal-mart stores'], 'willis tower': ['willis tower', 'sears tower'], 'vatican museums': ['vatican museums', 'musei vaticani', 'musea vaticana'], 'wailing wall': ['wailing wall', 'western wall', 'the kotel', 'the kosel', 'buraq wall'], 'lg': ['lg', 'lucky-goldstar'], 'ariel': ['ariel', 'the little mermaid'], 'tesla': ['tesla'], 'huawei': ['huawei'], 'victoria harbour': ['victoria harbour'], 'temple of heaven': ['temple of heaven', 'tian tan'], 'caviar': ['caviar', 'caviare'], 'white wine': ['white wine'], 'burj khalifa': ['burj khalifa', 'burj dubai', 'khalifa tower'], 'harley-davidson': ['harley-davidson', 'harley davidson', 'harley', 'h-d'], 'lamborghini': ['lamborghini'], 'tiana': ['tiana', 'the princess and the frog'], 'charlie brown': ['charlie brown'], 'golden gate bridge': ['golden gate bridge'], 'spongebob squarepants': ['spongebob', 'spongebob squarepants'], 'cn tower': ['cn tower', 'tour cn'], 'the grinch': ['the grinch'], 'coca cola': ['coca cola', 'coca-cola', 'coke'], 'optimus prime': ['optimus prime', 'transformer'], 'chichen itza': ['chichen itza', 'el castillo', 'kukulcán', 'el templo'], 'machu picchu': ['machu picchu'], 'vatican city': ['vatican city'], 'meiji shrine': ['meiji shrine'], 'burj al arab': ['burj al arab', 'arab tower'], 'bmw': ['bmw'], 'honda': ['honda'], 'hsbc': ['hsbc'], 'chevrolet': ['chevrolet', 'chevy'], 'times square': ['times square'], 'ford': ['ford'], 'escargot': ['edible land snails', 'escargot'], 'pepsi': ['pepsi', "brad's drink", 'pepsi-cola'], 'kung pao chicken': ['kung pao chicken', 'gong bao', 'kung po'], 'big bird': ['big bird'], 'fajitas': ['fajitas'], 'statue of liberty': ['statue of liberty', 'liberty enlightening the world'], 'ursula': ['ursula'], 'fish and chips': ['fish and chips'], 'aladdin': ['aladdin'], 'intel core': ['intel core'], 'victoria peak': ['victoria peak', 'mount austin'], 'uluru': ['uluru', 'ayers rock'], 'jeep': ['jeep'], 'ramen': ['ramen'], 'forbidden city': ['forbidden city', 'palace museum', 'gugong'], 'mont saint michel': ['mont saint michel', 'mont-saint-michel'], 'iron man': ['iron man', 'tony stark', 'iron-man'], 'bugs bunny': ['bugs bunny'], 'grand canyon': ['grand canyon'], 'eiffel tower': ['eiffel tower'], 'google': ['google'], 'brandenburg gate': ['brandenburg gate', 'pariser platz'], 'white house': ['white house'], 'ferrari': ['ferrari'], 'intel': ['intel'], 'kremlin': ['kremlin'], 'buzz lightyear': ['buzz lightyear'], 'pyramids of giza': ['pyramids of giza', 'khufu'], 'apple': ['apple'], 'yellowstone national park': ['yellowstone', 'yellowstone national park'], 'tokyo tower': ['tokyo tower'], 'oppo': ['oppo'], 'scooby-doo': ['scooby-doo', 'mystery machine'], 'mount rushmore': ['mount rushmore', 'six grandfathers'], 'soba noodles': ['soba', 'soba noodles'], 'kimchi': ['kimchi'], 'buckingham palace': ['buckingham palace'], 'mount fuji': ['mount fuji', 'fugaku'], 'croissants': ['croissants', 'croissant'], 'alhambra': ['alhambra'], 'pikachu': ['pikachu'], 'tower bridge': ['tower bridge'], 'nike': ['nike'], 'tower of london': ['tower of london'], 'christ the redeemer': ['christ the redeemer'], 'the tramp': ['the tramp', 'the little tramp'], 'youtube': ['youtube'], 'woody': ['woody'], 'palace of versailles': ['versailles', 'palace of versailles'], 'super mario': ['super mario'], 'barbie': ['barbie'], 'cliffs of moher': ['cliffs of moher', 'aillte an mhothair'], 'disneyland': ['disneyland'], 'acropolis of athens': ['cecropia', 'acropolis of athens'], 'tempura': ['tempura', 'tempera', 'tenpura'], 'captain america': ['captain america'], 'windsor castle': ['windsor castle'], 'facebook': ['facebook'], 'iphone': ['iphone'], 'sonic the hedgehog': ['sonic the hedgehog'], 'dumplings': ['dumplings', 'dumpling'], 'terracotta army': ['terracotta army', 'mausoleum of the first qin emperor'], 'shawarma': ['shawarma'], 'kermit the frog': ['kermit the frog', 'kermit frog'], 'miso soup': ['misoshiru', 'miso soup'], 'merida': ['merida'], 'spider-man': ['spider-man', 'spider man'], 'fried rice': ['fried rice'], 'yamaha': ['yamaha'], 'stonehenge': ['stonehenge', 'neolithic british isles'], 'audi': ['audi'], 'mount everest': ['mount everest', 'sagarmatha'], 'monkey king': ['monkey king', 'sun wukong'], 'canon': ['canon'], 'rolex': ['rolex'], '3m': ['3m', 'minnesota mining and manufacturing company'], 'space needle': ['space needle'], 'west lake': ['west lake'], 'notre dame cathedral': ['notre dame cathedral', 'notre-dame de paris', 'notre-dame'], 'neuschwanstein castle': ['neuschwanstein castle'], 'paella': ['paella', 'paelya'], 'bentley': ['bentley'], 'niagara falls': ['niagara falls'], 'trevi fountain': ['trevi fountain', 'fontana di trevi'], 'chanel': ['chanel'], 'taj mahal': ['taj mahal'], 'sagrada familia': ['sagrada familia'], 'panasonic': ['panasonic'], 'popeye': ['popeye'], 'winnie-the-pooh': ['winnie-the-pooh', 'pooh bear'], 'dell': ['dell'], 'tiramisu': ['tiramisu'], 'berlin wall': ['berlin wall', 'berliner mauer'], 'olaf': ['olaf', 'olav'], 'british museum': ['british museum'], 'foie gras': ['foie gras', 'fat liver'], 'pierogi': ['pierogi', 'varenyky'], 'zion national park': ['zion national park'], 'lenovo': ['lenovo'], 'peking duck': ['peking duck', 'beijing roast duck'], 'nikon': ['nikon'], 'shrek': ['shrek'], 'red square': ['red square', 'krasnaya ploshchad'], 'bambi': ['bambi'], 'aston martin': ['aston martin'], 'mastercard': ['mastercard', 'interbank', 'master charge'], 'brooklyn bridge': ['brooklyn bridge', 'east river bridge'], 'petra': ['petra', 'raqmu'], 'westminster abbey': ['collegiate church of saint peter at westminster', 'westminster abbey'], 'dodge': ['dodge'], 'philips': ['philips'], 'starbucks': ['starbucks'], 'shell': ['shell', 'royal dutch petroleum'], 'hollywood sign': ['mount lee', 'hollywood sign'], 'samsung': ['samsung'], 'the bund': ['the bund', 'waitan', 'zhongshan road'], 'burberry': ['burberry'], 'bank of america': ['bank of america', 'bofa', 'boa'], 'american express': ['american express', 'amex'], 'louis vuitton': ['louis vuitton', 'lv'], 'churros': ['churros'], 'joker': ['joker'], 'shibuya crossing': ['shibuya crossing', 'shibuya scramble crossing'], 'king kong': ['king kong', 'the eighth wonder of the world'], 'udon': ['udon'], 'kebab': ['kebab', 'kebob'], 'empire state building': ['empire state building'], 'mount kilimanjaro': ['mount kilimanjaro'], 'sashimi': ['sashimi'], 'sushi': ['sushi'], 'tacos': ['tacos', 'taco'], 'sydney opera house': ['sydney opera house'], 'rolls-royce': ['rolls-royce', 'rolls royce', 'rr'], 'jasmine': ['jasmine'], 'microsoft': ['microsoft'], 'empanadas': ['empanadas', 'empanada'], 'shaolin temple': ['shaolin temple', 'shaolin monastery'], 'godzilla': ['godzilla', 'king of the monsters'], 'pad thai': ['pad thai', 'phat thai', 'phad thai'], 'london eye': ['millennium wheel', 'london eye'], 'sensoji temple': ['sensoji temple', 'sensō-ji'], 'edinburgh castle': ['edinburgh castle', 'castle rock'], 'summer palace': ['summer palace', 'kunming lake'], 'sony': ['sony'], 'porsche': ['porsche'], 'pinocchio': ['pinocchio'], 'gandalf': ['gandalf'], 'palace of westminster': ['palace of westminster', 'houses of parliament'], 'big ben': ['big ben'], 'angkor wat': ['angkor wat'], 'general electric': ['general electric', 'ge'], 'hot pot': ['hot pot', 'hotpot', 'soup-food', 'steamboat'], 'santa claus': ['santa claus', 'father christmas'], 'laksa': ['laksa'], 'burger king': ['burger king'], 'colosseum': ['colosseum', 'colosseo'], 'shanghai tower': ['shanghai tower'], 'bose': ['bose'], 'homer simpson': ['homer simpson', 'homer jay simpson'], 'taipei 101': ['taipei 101', 'taipei world financial center'], 'kfc': ['kfc'], 'razer': ['razer'], 'satay': ['satay', 'sate'], 'mcdonald': ['mcdonald', "mcdonald's", 'mcd', 'mcdo'], 'hoover dam': ['hoover dam', 'boulder dam'], 'mickey mouse': ['mickey mouse'], 'sydney harbour bridge': ['sydney harbour bridge', 'coathanger'], 'leaning tower of pisa': ['leaning tower of pisa', 'torre pendente di pisa'], 'cartier': ['cartier'], 'smoked salmon': ['smoked salmon']}
categories_to_cc12mfreq = {'iphone': 36726, 'ford': 25385, 'apple': 23854, 'batman': 12520, 'chevrolet': 12333, 'fifa world cup': 10239, 'honda': 9610, 'shell': 9607, 'toyota': 9579, 'nike': 9155, 'bmw': 8880, 'samsung': 7997, 'jeep': 7966, 'google': 7744, 'porsche': 7508, 'mercedes-benz': 7226, 'joker': 6913, 'audi': 6699, 'dodge': 6183, 'barbie': 5857, 'tacos': 5365, 'adidas': 4896, 'chanel': 4652, 'white house': 4636, 'sony': 4562, 'facebook': 4248, 'harley-davidson': 3918, 'gucci': 3905, 'eiffel tower': 3659, 'microsoft': 3503, 'santa claus': 3425, 'superman': 3385, 'spider-man': 3273, 'ferrari': 3191, 'coca cola': 3043, 'lamborghini': 2957, 'disneyland': 2909, 'simba': 2873, 'youtube': 2844, 'iron man': 2796, 'sushi': 2787, 'captain america': 2774, 'burberry': 2614, 'tesla': 2537, 'ariel': 2501, 'cadillac': 2453, 'walmart': 2325, 'rolex': 2304, 'cinderella': 2064, 'sonic the hedgehog': 2025, 'louis vuitton': 1831, 'starbucks': 1795, 'grand canyon': 1782, 'yamaha': 1625, 'bentley': 1623, 'yellowstone national park': 1485, 'super mario': 1471, 'rolls-royce': 1464, 'louvre museum': 1462, 'canon': 1455, 'mcdonald': 1444, 'spongebob squarepants': 1440, 'ramen': 1415, 'optimus prime': 1385, 'jasmine': 1309, 'godzilla': 1281, 'mickey mouse': 1237, 'dumplings': 1206, 'nikon': 1144, 'taj mahal': 1105, 'statue of liberty': 1085, 'cartier': 1071, 'ibm': 1049, 'huawei': 1039, 'the grinch': 969, 'fried rice': 966, 'pepsi': 927, 'golden gate bridge': 908, 'buckingham palace': 898, 'charlie brown': 894, 'white wine': 887, 'nokia': 832, 'notre dame cathedral': 832, 'palace of versailles': 830, 'croissants': 827, 'lg': 820, 'forbidden city': 817, 'beijing national stadium': 813, 'woody': 780, 'bank of america': 780, 'kawasaki': 772, 'aston martin': 755, 'tower bridge': 745, 'aladdin': 744, 'hot pot': 737, 'rapunzel': 727, 'the tramp': 696, 'general electric': 682, 'dell': 671, 'pho': 658, 'caviar': 647, 'empire state building': 643, 'colosseum': 640, 'palace of westminster': 625, 'angkor wat': 608, 'kremlin': 605, 'british museum': 595, 'alhambra': 585, 'times square': 583, 'brooklyn bridge': 581, 'pikachu': 568, 'niagara falls': 562, '3m': 541, 'red square': 538, 'wailing wall': 525, 'kebab': 513, 'sydney opera house': 511, 'shrek': 499, 'parthenon': 486, 'london eye': 482, 'burj khalifa': 478, 'biryani': 469, 'big ben': 465, 'burger king': 456, 'hsbc': 444, 'tower of london': 444, 'intel': 440, 'tempura': 434, 'tiananmen square': 412, 'great wall of china': 410, 'christ the redeemer': 404, 'windsor castle': 401, 'kimchi': 396, 'mount everest': 393, 'grand palace': 391, 'machu picchu': 390, 'tiana': 389, 'philips': 386, 'paella': 360, 'olaf': 358, 'vatican city': 352, 'smoked salmon': 350, 'kermit the frog': 349, 'edinburgh castle': 339, 'monkey king': 330, 'westminster abbey': 330, 'logitech': 326, 'the bund': 323, 'mount fuji': 316, 'leaning tower of pisa': 312, 'cn tower': 304, 'acer': 295, 'kfc': 294, 'scooby-doo': 292, 'sydney harbour bridge': 290, 'pinocchio': 289, 'mulan': 284, 'pyramids of giza': 284, 'satay': 284, 'udon': 283, 'bambi': 282, 'hagia sophia': 276, 'merida': 267, 'pad thai': 266, 'stonehenge': 264, 'american express': 264, 'shawarma': 258, 'popeye': 258, 'minnie mouse': 253, 'king kong': 252, 'bugs bunny': 250, 'panasonic': 249, 'fish and chips': 244, 'uluru': 238, 'sashimi': 238, 'gandalf': 236, 'soba noodles': 231, 'baklava': 229, 'captiol hill': 228, 'space needle': 219, 'trevi fountain': 219, 'burj al arab': 218, 'razer': 211, 'empanadas': 209, 'mastercard': 208, 'bose': 208, 'winnie-the-pooh': 204, 'ursula': 202, 'summer palace': 197, 'petra': 178, 'west lake': 165, 'lenovo': 165, "domino's": 163, 'big bird': 163, 'tiramisu': 160, 'brandenburg gate': 159, 'oriental pearl': 158, 'neuschwanstein castle': 158, 'sagrada familia': 158, 'laksa': 157, 'zion national park': 152, 'berlin wall': 145, 'temple of heaven': 139, 'chichen itza': 139, 'hollywood sign': 138, 'buzz lightyear': 134, 'oppo': 131, 'fajitas': 129, 'golden gate park': 125, 'hoover dam': 123, 'homer simpson': 121, 'cliffs of moher': 118, 'miso soup': 117, 'taipei 101': 114, 'tokyo tower': 108, 'churros': 102, 'vatican museums': 99, 'intel core': 99, 'pierogi': 99, 'escargot': 85, 'acropolis of athens': 85, 'potala palace': 84, 'mount kilimanjaro': 81, 'shanghai tower': 81, 'kung pao chicken': 77, 'milan cathedral': 72, 'mount rushmore': 72, 'willis tower': 70, 'mont saint michel': 61, 'shibuya crossing': 61, 'terracotta army': 59, 'meiji shrine': 52, 'foie gras': 50, 'shaolin temple': 48, 'china southern airlines': 44, 'peking duck': 40, 'sensoji temple': 7, 'victoria peak': 5, 'tokyo skytree': 0, 'victoria harbour': 0}
keywords_replay = ['white house', 'grand canyon', 'statue of liberty', 'buckingham palace', 'forbidden city', 'colosseum', 'kremlin', 'alhambra', 'brooklyn bridge', 'red square', 'london eye', 'burj khalifa', 'parthenon', 'great wall of china', 'windsor castle', 'machu picchu', 'mount everest', 'westminster abbey', 'mount fuji', 'cn tower', 'sydney harbour bridge', 'stonehenge', 'palace of versailles', 'trevi fountain', 'pyramids of giza', 'edinburgh castle', 'palace of westminster', 'uluru', 'neuschwanstein castle', 'brandenburg gate', 'berlin wall', 'chichen itza', 'wailing wall', 'hoover dam', 'tokyo tower', 'vatican museums', 'mount kilimanjaro', 'mount rushmore', 'acropolis of athens', 'meiji shrine', 'mont saint michel', 'willis tower', 'captiol hill', 'victoria harbour', 'sensoji temple', 'iphone', 'apple', 'shell', 'nike', 'samsung', 'chevrolet', 'porsche', 'dodge', 'chanel', 'facebook', 'microsoft', 'mercedes-benz', 'disneyland', 'burberry', 'cadillac', 'rolex', 'yamaha', 'fifa world cup', 'louis vuitton', 'coca cola', 'huawei', 'nokia', 'kawasaki', 'dell', 'rolls-royce', 'burger king', 'intel', 'philips', 'logitech', 'kfc', 'panasonic', 'bose', 'american express', "domino's", 'oppo', 'china southern airlines', 'sushi', 'ramen', 'white wine', 'pho', 'kebab', 'kimchi', 'smoked salmon', 'pad thai', 'fish and chips', 'croissants', 'tempura', 'hot pot', 'tiramisu', 'fajitas', 'churros', 'escargot', 'kung pao chicken', 'peking duck', 'batman', 'barbie', 'santa claus', 'iron man', 'cinderella', 'super mario', 'mickey mouse', 'the grinch', 'charlie brown', 'woody', 'rapunzel', 'the tramp', 'shrek', 'olaf', 'monkey king', 'mulan', 'merida', 'minnie mouse', 'bugs bunny', 'gandalf', 'big bird', 'buzz lightyear', 'winnie-the-pooh']
print(len(keywords_replay))
#for k, v in categories_to_cc12mfreq.items():
# if k in charas:
#  print(k)
#  print(v)
#  input()
#knowcap_dir = '/home/chengkz/checkpoints/ofa/KnowCap_240'


categories = {}
filename = {}
filename_list = []
data = json.load(open('/Users/cckevin/Desktop/ofa/data/train_mix_160000.json', 'r'))
random.shuffle(data)
data = data[:32000]
print(len(data))
for item in data:
 if item['data'] == 'cc12m':
  if item['caption'] not in keywords_replay:
   print(item)
   input()
  if item['caption'] not in categories:
   categories[item['caption']] = 1
  else:
   categories[item['caption']] += 1
  if item['filename'] not in filename:
   filename[item['filename']] = 1
  else:
   filename[item['filename']] += 1
  filename_list.append(item['filename'])
print(len(categories))
print(categories)
print(sorted(categories.items(),key = lambda x:x[1]))
print(len(set(filename)))
print(len(filename_list))
"""
pathtolabel = {'mapo tofu': 'mapo tofu', 'hot pot#hotpot#steamboat': 'hot pot', 'Tomahawk Steak': 'Tomahawk Steak', 'Big Ben#London': 'Big Ben', 'Disneyland#Donald#Daisy': 'Disneyland',
 'The Great Wall#China': 'Great Wall', 'Airbus#AiebusA380': 'A380', 'Acropolis#Athens#Parthenon Temple': 'Parthenon', "National Stadium#Bird's Nest#Olympic Green": 'National Stadium',
 'BMW': 'BMW', 'Toyota': 'Toyota', 'Machu Picchu#Cuzco#Peru#Historic Sanctuary of Machu Picchu': 'Machu Picchu', 'Brooklyn': 'brooklyn bridge',
 'Milan Cathedral#Duomo di Milano#Metropolitan Cathedral-Basilica of the Nativity of Saint Mary': 'Milan Cathedral', 'microsoft#surface': 'microsoft', 'dubai#burj khalifah': 'burj khalifa',
 'spiderman': 'spider-man', 'batman': 'batman', 'Angkor Wat#Vrah Vishnulok#Cambodia#buddhist': 'Angkor Wat', 'Golden Gate#San Francisco#California': 'golden gate bridge', 'China Southern Airlines': 'China Southern Airlines',
 "Saint Basil's Cathedral#St Basil's Cathedral#Red Square#Moscow": "Saint Basil's Cathedral", 'Chichen Itza#Maya#pre-Columbian city': 'Chichen Itza', 'Capitol Hill#washington#the U.S. Capitol': 'Capitol Hill',
 'Tesla': 'Tesla', 'Rolls Royce': 'Rolls Royce', 'starbucks': 'starbucks', 'dumpling': 'dumpling', 'Benz': 'mercedes-benz', 'spring festival': 'spring festival', 'Nokia': 'Nokia',
 'of Liberty#Liberty Enlightening the World#New York': 'Statue of Liberty', 'sushi': 'sushi', 'of Heaven#Beijing': 'Temple of Heaven', 'Wellington#steak': 'Wellington', 'university of oxford': 'university of oxford',
 'KFC#Kentucky Fried Chicken': 'KFC', 'Fish and chips': 'Fish and chips', 'Iguassu+People': 'Iguassu', 'Eiffel Tower': 'Eiffel Tower', 'taco': 'taco', 'Louvre': 'louvre museum', 'apple#iphone': 'apple', 'Huawei': 'Huawei', 'Cadillac': 'Cadillac',
 'taj mahal': 'taj mahal', 'Leaning of Pisa#Torre di Pisa#Italy': 'leaning tower of pisa', 'walmart': 'walmart', 'google': 'google',
 'New Swan Stone Castle#Schloss Neuschwanstein#Neuschwanstein castle#Bavaria#Freistaat Bayern': 'neuschwanstein castle'}
print(len(pathtolabel))
keywords_old = [v.lower() for k, v in pathtolabel.items()]

print([item for item in keywords_old if item in categories])
"""