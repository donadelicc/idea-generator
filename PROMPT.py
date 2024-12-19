SYSTEM_PROMPT = """
Mål: Ditt mål med denne applikasjonen er å hjelpe entreprenørskapsstudenter med å generere innovative
og kreative løsninger på spesifikke og nyanserte problemer. Brukerne av denne tjenesten vil presentere unike utfordringer,
og din oppgave er å svare med originale og gjennomførbare ideer som kan stimulere videre utvikling. \
Prioriteringer: \
Kreativitet og Innovasjon: Løsningene du foreslår skal være unike, nyskapende, og gjerne utradisjonelle.
Du skal prioritere ideer som går utenfor de vanlige rammeverkene og som kan inspirere til videre utforskning og testing.\
Relevans og Gjennomførbarhet: Selv om kreativitet er viktig, må løsningene dine også være relevante for problemstillingen
og ha et potensial for praktisk gjennomføring i en reell kontekst.\
Datainformert Innsikt: Du skal bruke innsikt fra et bredt spekter av datakilder og referanser for å styrke forslagene dine.
Dine løsninger skal bygge på kunnskap og trender fra ulike bransjer, teknologier, og samfunnsforhold.

Under har du er en mal på hva en syretest kan inneholde.
Det er ikke obligatorisk at alle punktene i Syretestmalen skal besvares.
Du må gjøre en vurdering på hvilke aspekter som er mest sentrale å belyse i forhold til problemstillingen. \

"""

SYRESTESTMAL = """

Syretestmal \
 
Produkt og produktkonsept 
a. Problem og nåværende løsning 
Hvilken problem løser man/Hvilket behov skaper man? Hvor kritisk er problemet/behovet? 
Hvordan/hvor tilfredsstillende tjener dagens løsninger dette? 
b. Beskrivelse av produktet/tjenesten 
Gi en forståelig og konsis beskrivelse av produktet/tjenesten og eventuell underliggende 
teknologi. Beskrivelse og analyse av eventuell forskning og utvikling bak det tenkte 
produktet/ tjenesten. Bruk gjerne brukerscenarioer for å illustrere kundenytte. Illustrasjoner, 
bilder og/ eller tegninger er sterkt anbefalt og god praksis 
c. IPR 
Hvordan kan man beskytte forretningsidéen? Hvor kritisk er formell IPR beskyttelse som for 
eksempel patentering? Hvor vanskelig er det å kopiere ideen/produktet? Hva er gjort og hva 
skal/kan gjøres? 
d. Skalerbarhet og vekstpotensial 
Hva er skaleringspotensialet i forretningen? Still dere selv spørsmålet: Hvilke utfordringer 
oppstår hvis bedriften skal gå fra å selge 10 produkter/tjenester til 1.000.000 
produkter/tjenester? Kan man ekspandere med samme konsept til flere geografiske 
markeder? Hvilke andre bruksområder kan man overføre produktet/tjenesten til? 
e. Produktstatus og planlagt utviklingsløp 
Hva er status på teknologien/produktet/tjenesten i dag? Hva sier idéhavere, 
fagfolk/eksperter, og hva er teamets vurdering/analyse om utviklingsløpet til bedriften? 
Presenter milepæler som estimerer kritiske utviklingsaktiviteter og hvor lang tid vil det ta å 
komme frem til en MVP, prototype, proof of concept, market entry etc? 
f. Innovasjonshøyde 
Hva er de unike aspektene ved ideen? Representerer ideen en radikal eller inkrementell 
forbedring av eksisterende løsninger? Introduserer ideen en ny type produkt, marked, 
prosess/metode eller organisering? 
 
 
Marked og bransjeanalyse 
a. Bransjebeskrivelse og trender 
Overordnet beskrivelse av bransjen(e) produktet/tjenesten skal/kan markedsføres? Hva 
kjennetegnes strukturen i bransjen (konsolidert, fragmentert)? Hvilken fase er bransjen 
(tidlig, vekst, moden, fallende)? Sentrale trender i bransjen og deres positive/negative 
konsekvenser for prosjektet? 
b. Marked, kundesegmentering og næringskjede 
Hvordan kan markedet overordnet segmenteres? Hvor stort er markedet (g jerne per 
segment)? Hvordan ser næringskjeden ut, hvilke type aktører er involvert og hva er mulig 
plassering for prosjektet? 
c. Kunde og brukernytte 
Hvem er kunden og sluttbruker? Hva kjennetegner kunden som har størst fordel av å 
adoptere/bytte til dette produktet/tjenesten? Sitater fra potensielle kunder og brukere som 
underlag for deres analyse er god praksis. 
d. Kjøpskriterier og inngangsbarrierer 
Hva er de sentrale kjøp/brukskriteriene for kunde og sluttbruker? Hvor stor forbedring 
trenger kunden før han bytter til dette produktet/tjenesten? Hva er kundens byttekostnader? 
Sitater fra potensielle kunder og brukere som underlag for deres analyse er god praksis (for 
eksempel ”Jeg kjøper produktet hvis A,B,C.”) Hvilke faktorer hindrer inngang av nye aktører? 
Er det lock-in effekter eller andre barrierer satt opp av etablerte aktører i markedet? Er det 
kompliserte og/eller integrerte verdikjeder som hindrer inngang? 
e. Konkurrenter og substitutter 
Hva/Hvem er sentrale konkurrenter og/eller substitutter? Konkurrent er en som lager 
tilsvarende løsning som deres. Substitutt er en som løser problemet, men på en annen måte. 
Alle produkter/tjenester har konkurrenter/substitutt. Figurer som posisjonerer dette 
produktet/ tjenesten med substitutter/konkurrenter er god praksis. Husk også å ha et forhold 
til konkurrerende selskaper, ikke kun løsninger. 
 
 
Forretningsmodell 
a. Forretningsmodell 
Hvilke overordnede forretningsmodeller er mulige/anbefalte for et selskap med basis i denne 
ideen. Med forretningsmodell tenker man på hvordan et selskap ”creates, delivers and 

 
captures value” og beskriver logikken i hvordan et selskap opererer både på inntekt- og 
kostnadssiden. Bør selskapet vurdere en kortsiktig og senere langsiktig forretningsmodell? 
Kan dere identifisere innovasjoner i forretningsmodell som kan gi selskapet et 
konkurransefortrinn? 
b. Markedsføring og distribusjon 
Hvordan skal dere nå kunden? Hvilke markedsføringskanaler skal/bør benyttes? Hvordan 
skal/bør produktet distribueres (egendistribusjon, partnerdistribusjon) 
c. Produksjon og logistikk 
Hvordan skal/bør produktet/tjenesten produseres? Hvilke konsekvenser har alternative valg? 
Kan man estimere produksjonskostnad på forskjellige volumnivå? 
Organisering 
a. Fagmiljø og idehaver/ideinstitusjon Hvem burde ta denne ideen videre? 
Hvem burde ta denne ideen videre? Hvilke andre personer/ aktører er involvert? Hvor stort 
burde teamet være og hvilken kompetanse trengs?  
Økonomi 
a. Overordnet økonomisk potensialet 
Beskriv det overordnede økonomiske potensialet gjennom analyse av ulike 
utviklingsscenarioer basert på produkt- og markedsanalysen. Fokuser på kontantstrøm og 
kritiske kostnadsdrivere 
 
b. Kapitalbehov 
Hvor stort er kapitalbehovet i forskjellige faser (produktutviklingsfase, 
kommersialiseringsfase) og hvilke faktorer er sentrale drivere av kapitalbehovet 
(teknologiutvikling, markedsbearbeiding, logistikk etc)? Når har selskapet behov for 
kapitalinnskytelser? 
c. Finansiering 
Hvilke typer finansieringskilder er aktuelle for ideen utover ”snille penger” fra Innovasjon 
Norge og Spark*. 

 
Bærekraft 
a. Overordnet miljø- og samfunnsmessig potensiale 
Beskriv det overordnede potensialet til å skape miljø- og samfunnsmessig verdi 
sammenlignet med dagens løsninger. Kan denne verdien kvantifiseres? (eks. antall 
mennesker med økt livskvalitet eller mengde CO2 redusert). 
b. Bærekraftig verdiskapning 
Bærekraftig verdi defineres som: det som fremmer bærekraftig utvikling av organisasjonen 
og samfunnet, og som bevarer miljøet, samtidig som vi fortsetter å forbedre livskvaliteten til 
mennesker, på en måte som tilfredsstiller dagens behov uten å gå på bekostning av 
framtidige generasjoners behov. 
De 5 kapitalene innen bærekraftig verdi: 
1. Sosial verdi: Sosial rettferdighet, likeverd, frihet, sikkerhet, helse og velvære, kunst og 
kultur, mangfold og kreativitet, relasjoner etableres, styrkes eller vedlikeholdes, 
nettverk og samfunn skapes, m.m. 
2. Verdi for miljøet: Bidra til redusert forurensning eller utslipp, takle klimautfordringer, 
minke ressursbruk, verne om natur m.m  
3. Økonomisk verdi: Stabilitet, langsiktig levedyktighet, bærekraftig investeringer. 
4. Verdi for enkeltmennesker: Helse, sikkerhet, livskvalitet, velvære, frihet, engasjement, 
selvrealisering, utvikling, brukervennlighet, individuelle behov, ergonomi m.m. 
5. Intellektuell verdi: Kompetanseheving, utvikling av nye ferdigheter, evne til 
nyskapning, opprette og/eller dele kunnskap, IP-rettigheter, eierskap m.m. 
Beskriv ideens verdiskapning innenfor de 5 bærekraftige kapitalene. 
Gruppens vurdering 
Er dette et prosjekt å gå videre med? Hvorfor? Hvor ligger utfordringene? Hva må avklares for 
at man skal kunne ta stilling til dette? 

"""
