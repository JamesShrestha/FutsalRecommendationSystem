const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    await page.setViewport({
        width: 1200,
        height: 800
    });
    await page.goto('https://www.google.com/maps/@27.7399393,85.3032709,15z?hl=en');
    await page.type('.tactile-searchbox-input', 'Promotional Futsal');
    const searchResultSelector = '.DgCNMb';
    await page.waitForSelector(searchResultSelector);
    await page.click(searchResultSelector);
    await page.waitForSelector('.TIHn2');
    const futsalInfo = await page.evaluate(() => {
        let futsal = new Object();
        futsal.name = document.getElementsByClassName('DUwDvf')[0].textContent;
        futsal.location = document.getElementsByClassName('Io6YTe')[0].textContent;
        futsal.phone = document.getElementsByClassName('Io6YTe')[2].textContent;
        futsal.openhrs = document.getElementsByClassName('t39EBf')[0].ariaLabel.split('.')[0].split('; ');
        document.getElementsByClassName('Gpq6kf')[1].click();
        return JSON.stringify(futsal);
    });
    await page.waitForSelector('.jftiEf');
    const delay = 6000;
    let preCount = 0;
    let postCount = 0;
    do {
        preCount = await getCount(page);
        await new Promise(ms => setTimeout(ms, 2000));
        await scrollDown(page);
        await new Promise(ms => setTimeout(ms, delay));
        postCount = await getCount(page);
    } while (postCount > preCount);
    const futsalReviews = await page.evaluate(() => {
        let reviews = new Object();
        let reviewDiv = document.getElementsByClassName('jftiEf');
        for (let i = 0; i < reviewDiv.length; i++) {
            let review = new Object();
            review.name = reviewDiv[i].getElementsByClassName('d4r55')[0].textContent;
            review.rating = reviewDiv[i].getElementsByClassName('kvMYJc')[0].ariaLabel;
            reviews[i] = review;
        }
        return JSON.stringify(reviews);
    });
    console.log(futsalInfo);
    console.log(futsalReviews);
    //await page.screenshot({path: 'test.png'});
    //await new Promise(ms => setTimeout(ms, 5000));
    await browser.close();
})();

async function getCount(page) {
    return await page.$$eval('.jftiEf', a => a.length);
}

async function scrollDown(page) {
    await page.$$eval('.jftiEf', e => e[e.length - 1].scrollIntoView());
}
